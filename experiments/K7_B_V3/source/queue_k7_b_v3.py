#!/usr/bin/env python3
"""Detached K7-B V3 orchestrator.

This process owns the complete remote lifecycle: wait for verified v0921
parts, build V3 supplement and caches, run the K7-B preflight, then train to
20k with checkpoint-based recovery.  It does not depend on the SSH session.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("/root/projects/hrfont")
CODE = Path("/root/projects/hrfont_k7_v3_20260922")
TORCHRUN = str(Path(sys.executable).with_name("torchrun"))
RAW = PROJECT / "data/v0921_raw_20260921"
PARTS = RAW / "source_parts"
EXTRACT = RAW / "extracted"
V0921 = PROJECT / "data/v0921_v3_supplement_20260922"
V3 = PROJECT / "data/v3_v2_plus_v0921_20260922"
DATA = Path("/root/data1/hrfont_dataset_v2_20260917")
STORE = Path("/root/data1/hrfont_k6_20260920")
BASE_CKPT = PROJECT / "runs/F0-CLEAN-V0913-A-S3407/global_step_10000"
BASE_DATA = PROJECT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
BASE_ES = PROJECT / "artifacts/g0/es_spatial"
BASE_EC = PROJECT / "artifacts/g0/ec_multiscale"
SPEC = V3 / "v0921_spec.json"
RUN_ID = "K7-B-V3-K0-S3407-20K-R2"
CHECKPOINT_ROOT = V3 / "checkpoint_store"
LOG = V3 / "K7_ORCHESTRATOR.jsonl"
COMMIT = "c79ea9541f82bdd6f43c2c212b25bbfeccb0d12f"
PART_HASHES = {
    "texiao_pass1_319.part1of4.bin": (1365601041, "811a47113da6c20e355ed7dd7c36273a9e1ef4e9ad7cdc22c5289e62f719acc2"),
    "texiao_pass1_319.part2of4.bin": (1365601041, "fa019e0225fdf8113d5fa72bfc25d0b26a491dfa1da1ce66bd3f14333765f408"),
    "texiao_pass1_319.part3of4.bin": (1365601041, "010e5f4ac879cb47a70ced94cdba3d2b7f36881800c8906553477b4576ce5e37"),
    "texiao_pass1_319.part4of4.bin": (1365601038, "1bca2da6b02fd08f6c842ce2d212e4dcdfe471b307e885d6358a33d05346465a"),
}
DATA_PLAN = json.loads((PROJECT / "manifests/charset_cn2west_v2_planned.json").read_text(encoding="utf-8"))
TARGET_BUCKETS = DATA_PLAN["target"]
STYLE_HAN = DATA_PLAN["style_han_338"]
STYLE_CODES = [f"u{ord(ch):04X}" for ch in STYLE_HAN]
DATASET_BUILD_ID = "k7-v0921-v2-style-han-338-random-nshot-all-available-targets-v1"
SCRIPT_TO_BUCKETS = {"latin": ["ascii_digits", "ascii_letters"], "latin_ext": ["latin_ext_letters"], "hiragana": ["hiragana"], "katakana": ["katakana"], "zhuyin": ["bopomofo"]}


def log(event, **payload):
    V3.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.time(), "event": event, **payload}, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def dataset_is_current():
    try:
        marker = json.loads((V0921 / "BUILD_COMPLETE.json").read_text(encoding="utf-8"))
        pools = json.loads((V0921 / "style_pool.json").read_text(encoding="utf-8"))
        return (marker.get("build_id") == DATASET_BUILD_ID and len(pools) == 315
                and all(set(refs) == set(STYLE_CODES) for refs in pools.values()))
    except Exception:
        return False


def run(cmd, **kwargs):
    log("command_start", command=[str(x) for x in cmd])
    p = subprocess.run(cmd, text=True, **kwargs)
    log("command_done", returncode=p.returncode, command=[str(x) for x in cmd])
    if p.returncode: raise RuntimeError(f"command failed {p.returncode}: {cmd}")


def wait_parts():
    while True:
        ready = True
        for name, (size, digest) in PART_HASHES.items():
            path = PARTS / name
            if not path.is_file() or path.stat().st_size != size:
                ready = False; break
        if ready:
            for name, (_, digest) in PART_HASHES.items():
                path = PARTS / name
                got = sha256(path)
                if got != digest:
                    log("part_hash_mismatch", file=name, expected=digest, got=got); ready = False; break
        if ready:
            log("raw_parts_verified", parts=list(PART_HASHES)); return
        log("waiting_raw_parts")
        time.sleep(60)


def extract_source():
    # Once V3 and its caches are complete, the raw TTF extraction is no longer
    # needed by the training process.  This also makes detached recovery
    # idempotent after the temporary extraction directory is cleaned up.
    if dataset_is_current() and (V3 / "cache/ec/COMPLETE.json").is_file():
        log("source_extract_skipped_v3_complete")
        return
    archive = RAW / "texiao_pass1_319.joined.bin"
    if not archive.is_file() or archive.stat().st_size != sum(x[0] for x in PART_HASHES.values()):
        with archive.open("wb") as out:
            for name in PART_HASHES: out.write((PARTS / name).read_bytes())
        log("archive_joined", bytes=archive.stat().st_size, sha256=sha256(archive))
    marker = EXTRACT / "EXTRACT_COMPLETE.json"
    if marker.is_file(): return
    tmp = EXTRACT.with_name(EXTRACT.name + ".tmp")
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z: z.extractall(tmp)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t: t.extractall(tmp)
    else:
        raise RuntimeError("joined v0921 source is neither zip nor tar; inspect archive before continuing")
    tmp.rename(EXTRACT)
    marker.write_text(json.dumps({"status": "COMPLETE", "archive_sha256": sha256(archive)}, indent=2) + "\n")
    log("source_extracted", root=str(EXTRACT))


def metadata():
    dest = V3 / "metadata"
    dest.mkdir(parents=True, exist_ok=True)
    fonts_path = dest / "fonts_all_315.json"
    if not fonts_path.is_file():
        raw = subprocess.check_output(["git", "-C", str(PROJECT), "show", COMMIT + ":manifests/v0921_texiao_supplement/fonts_all_315.json"])
        fonts_path.write_bytes(raw)
    return json.loads(fonts_path.read_text(encoding="utf-8"))


def find_fonts(meta):
    by_name = {p.name: p for p in EXTRACT.rglob("*") if p.is_file()}
    # The raw archive keeps the original stem twice to avoid collisions, e.g.
    # UN2011974__UN2011974.TTF, while the frozen Git manifest uses UN2011974.TTF.
    for path in list(by_name.values()):
        if "__" in path.name:
            suffix = path.name.split("__", 1)[1]
            by_name.setdefault(suffix, path)
    missing = [r["font_file"] for r in meta["train"] if r["font_file"] not in by_name]
    if missing: raise RuntimeError(f"missing extracted TTF/OTF files: {missing[:10]} ({len(missing)})")
    return {r["clean"]: by_name[r["font_file"]] for r in meta["train"]}


def render_one(job):
    clean, font_path, scripts, missing_chars = job
    out = V0921 / "train"
    usable = []
    for script in scripts:
        for bucket in SCRIPT_TO_BUCKETS[script]:
            usable.extend(ch for ch in TARGET_BUCKETS[bucket] if ch not in missing_chars)
    usable = list(dict.fromkeys(usable))
    probe = ImageFont.truetype(str(font_path), 40)
    def okay(ch):
        try: return probe.getmask(ch).getbbox() is not None
        except Exception: return False
    usable = [ch for ch in usable if okay(ch)]
    refs = STYLE_HAN
    unsupported_refs = [ch for ch in refs if not okay(ch)]
    if unsupported_refs: raise RuntimeError(f"{clean} missing standard V2 style references: {unsupported_refs}")
    if not usable: raise RuntimeError(f"{clean} has no usable selected-script target glyphs")
    def save(ch, folder):
        path = out / folder / clean / f"{clean}+u{ord(ch):04X}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (96, 96), "white"); draw = ImageDraw.Draw(img)
        f = ImageFont.truetype(str(font_path), 80)
        box = draw.textbbox((0, 0), ch, font=f); w, h = box[2]-box[0], box[3]-box[1]
        draw.text(((96-w)//2-box[0], (96-h)//2-box[1]), ch, font=f, fill="black")
        img.save(path)
    for ch in refs: save(ch, "StyleImage")
    # The manifest's selected scripts and missing_chars define content targets.
    # References are the independent, bank-compatible V2 Han style pool.
    targets = usable
    for ch in targets: save(ch, "TargetImage")
    return clean, refs, targets


def build_dataset(meta):
    marker = V0921 / "BUILD_COMPLETE.json"
    if marker.is_file():
        pool_path = V0921 / "style_pool.json"
        pool = json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.is_file() else {}
        if dataset_is_current(): return
        # Recover the orchestrator's own incomplete/invalid build only.
        shutil.rmtree(V0921)
    if V0921.exists(): shutil.rmtree(V0921)
    V0921.mkdir(parents=True)
    font_paths = find_fonts(meta)
    jobs = []
    for row in meta["train"]:
        miss = set("".join((row.get("missing_chars") or {}).values()))
        jobs.append((row["clean"], font_paths[row["clean"]], row["scripts"], miss))
    results = []
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as ex:
        futures = [ex.submit(render_one, job) for job in jobs]
        for future in as_completed(futures): results.append(future.result())
    pools = {clean: [f"u{ord(ch):04X}" for ch in refs] for clean, refs, _ in results}
    pair_rows = []
    for clean, refs, targets in sorted(results):
        for ch in targets:
            cp = f"u{ord(ch):04X}"
            pair_rows.append(("train", clean, cp))
            content = V0921 / "train" / "ContentImage" / f"{cp}.png"
            source = DATA / "v2" / "train" / "ContentImage" / f"{cp}.png"
            content.parent.mkdir(parents=True, exist_ok=True)
            if not content.exists(): content.symlink_to(source)
    pairs = V0921 / "pairs_train.tsv"
    pairs.write_text("split\tfont\tcp\n" + "\n".join("\t".join(r) for r in pair_rows) + "\n", encoding="utf-8")
    pool_path = V0921 / "style_pool.json"; pool_path.write_text(json.dumps(pools, ensure_ascii=False, indent=2) + "\n")
    V3.mkdir(parents=True, exist_ok=True)
    spec = {"schema_version": 1, "dataset_id": "v0921", "combined_dataset_id": "v3_v2_plus_v0921", "role": "train_only_nonbank", "bank_inclusion": False, "split": "train", "data_root": str(V0921), "pairs": str(pairs), "style_pool": str(pool_path), "es_cache": str(V3 / "cache/es"), "ec_cache": str(V3 / "cache/ec"), "fonts": sorted(pools)}
    SPEC.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    base_pairs = sum(1 for _ in (DATA / "manifests/v2/pairs_train.tsv").open(encoding="utf-8")) - 1
    V3.mkdir(parents=True, exist_ok=True)
    (V3 / "manifest.json").write_text(json.dumps({"dataset_id": "v3_v2_plus_v0921", "status": "frozen", "base": "v2", "base_train_pairs": base_pairs, "supplement": "v0921", "supplement_fonts": len(pools), "supplement_train_pairs": len(pair_rows), "reference_protocol": "V2 style_han_338; random n-shot 1-8 via existing K sampler", "content_protocol": "all selected-script target buckets per frozen v0921 availability/missing_chars manifest", "val_test": "frozen V2 unchanged", "bank": "frozen V2 donor_train_by_cp only; v0921 never enters bank", "source_commit": COMMIT}, indent=2) + "\n")
    (V0921 / "BUILD_COMPLETE.json").write_text(json.dumps({"status": "COMPLETE", "build_id": DATASET_BUILD_ID, "fonts": len(pools), "style_refs_per_font": len(STYLE_CODES), "pairs": len(pair_rows)}, indent=2) + "\n")
    # Keep the verified source parts, but remove the extracted temporary copy
    # after the V3 supplement has been materialized.
    if EXTRACT.exists(): shutil.rmtree(EXTRACT)
    log("v3_dataset_built", fonts=len(pools), supplement_pairs=len(pair_rows), spec=str(SPEC))


def build_caches():
    # The legacy runtime verifies the frozen base cache against the exact K0
    # parent.  The old g0 cache was made from a removed checkpoint, so it must
    # be rebuilt before K7-B can even enter preflight.
    expected_es = sha256(BASE_CKPT / "style_encoder.pth")
    expected_ec = sha256(BASE_CKPT / "content_encoder.pth")
    def valid(path, key):
        try:
            man = json.loads((path / "manifest.json").read_text())
            prog = json.loads((path / "progress.json").read_text())
            return man.get(key) == (expected_es if key == "es_checkpoint_sha256" else expected_ec) and prog.get("done") == prog.get("total")
        except Exception:
            return False
    if not valid(BASE_ES, "es_checkpoint_sha256"):
        old = (json.loads((BASE_ES / "manifest.json").read_text()) if (BASE_ES / "manifest.json").is_file() else {})
        if BASE_ES.exists(): shutil.rmtree(BASE_ES)
        log("stale_base_es_cache_removed", old_manifest=old, expected=expected_es)
        run([sys.executable, str(CODE / "scripts/hrfont_build_e1_caches.py"), "--which", "es", "--gpu", "0", "--ckpt-dir", str(BASE_CKPT), "--data-root", str(BASE_DATA), "--split", str(PROJECT / "manifests/split_v3_228_16_16.json"), "--es-out", str(BASE_ES), "--batch-size", "16"], cwd=str(CODE), env=dict(os.environ, PYTHONPATH=f"{CODE}:{CODE / 'scripts'}"))
    if not valid(BASE_EC, "ec_checkpoint_sha256"):
        old = (json.loads((BASE_EC / "manifest.json").read_text()) if (BASE_EC / "manifest.json").is_file() else {})
        if BASE_EC.exists(): shutil.rmtree(BASE_EC)
        log("stale_base_ec_cache_removed", old_manifest=old, expected=expected_ec)
        run([sys.executable, str(CODE / "scripts/hrfont_build_e1_caches.py"), "--which", "ec", "--gpu", "0", "--ckpt-dir", str(BASE_CKPT), "--data-root", str(BASE_DATA), "--split", str(PROJECT / "manifests/split_v3_228_16_16.json"), "--ec-out", str(BASE_EC), "--batch-size", "8"], cwd=str(CODE), env=dict(os.environ, PYTHONPATH=f"{CODE}:{CODE / 'scripts'}"))
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    pools = json.loads(Path(spec["style_pool"]).read_text(encoding="utf-8"))
    pair_lines = [line for line in Path(spec["pairs"]).read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    expected_entries = {
        "es": sum(len(refs) for refs in pools.values()),
        "ec": len({line.split("\t")[2] for line in pair_lines}) + len(pair_lines),
    }
    def extra_cache_valid(kind):
        path = V3 / "cache" / kind
        try:
            complete = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
            ident = complete["identity"]
            return (complete.get("status") == "COMPLETE" and ident.get("dataset_id") == "v0921"
                    and ident.get("bank_inclusion") is False and ident.get("entries") == expected_entries[kind]
                    and ident.get("keys_sha256") == sha256(path / "keys.txt"))
        except Exception:
            return False
    if extra_cache_valid("es") and extra_cache_valid("ec"):
        log("v3_caches_verified", entries=expected_entries)
        return
    cmd = [sys.executable, str(CODE / "experiments/K6/implementation/r3/scripts/build_v0921_caches.py"), "--spec", str(SPEC), "--ckpt", str(PROJECT / "runs/F0-CLEAN-V0913-A-S3407/global_step_10000"), "--es-out", str(V3 / "cache/es"), "--ec-out", str(V3 / "cache/ec"), "--gpu", "0", "--batch", "16"]
    env = dict(os.environ, PYTHONPATH=f"{CODE}:{CODE / 'scripts'}")
    run(cmd, env=env)
    spec = json.loads(SPEC.read_text()); spec["es_cache"] = str(V3 / "cache/es"); spec["ec_cache"] = str(V3 / "cache/ec"); SPEC.write_text(json.dumps(spec, indent=2) + "\n")
    log("v3_caches_built")


def identity(): return json.loads((CODE / "K7_CODE_IDENTITY.json").read_text())


def preflight():
    pf = STORE / "control/PREFLIGHT_PASSED_K7_B.json"
    if pf.is_file(): return
    for attempt in range(9, 29):
        run_dir = f"K7-B-PREFLIGHT-V3-S3407-R{attempt}-K6-REFS"
        cmd = [TORCHRUN, "--standalone", "--nproc_per_node=8", str(CODE / "experiments/K6/implementation/r3/scripts/train_k6.py"), "--arm", "K7-B", "--run-id", run_dir, "--limit", "2", "--smoke", "--state-interval", "2", "--v0921-spec", str(SPEC)]
        try:
            run(cmd, cwd=str(CODE), env=dict(os.environ, PYTHONPATH=f"{CODE}:{CODE / 'scripts'}"))
        except Exception as exc:
            log("preflight_retry", attempt=attempt, run_id=run_dir, error=str(exc))
            time.sleep(min(300, 15 * (attempt - 8)))
            continue
        payload = {"status": "PASS", "arm": "K7-B", "identity": identity(), "checks": ["8-rank forward/backward", "independent K0 load", "V3 sampler", "v0921 bank exclusion", "offset loss weight zero"], "v3_manifest": str(V3 / "manifest.json"), "preflight_run_id": run_dir, "time": time.time()}
        pf.write_text(json.dumps(payload, indent=2) + "\n")
        log("k7_preflight_passed", run_id=run_dir)
        return
    raise RuntimeError("K7-B preflight exceeded recovery retry budget")


def train():
    run_root = PROJECT / "runs" / RUN_ID
    if (run_root / "DONE.json").is_file(): return
    common = [TORCHRUN, "--standalone", "--nproc_per_node=8", str(CODE / "experiments/K6/implementation/r3/scripts/train_k6.py"), "--arm", "K7-B", "--run-id", RUN_ID, "--limit", "20000", "--state-interval", "1000", "--checkpoint-root", str(CHECKPOINT_ROOT), "--v0921-spec", str(SPEC)]
    for attempt in range(1, 21):
        resume = run_root / "last_state"
        cmd = list(common)
        if resume.is_dir(): cmd += ["--resume", str(resume)]
        try:
            run(cmd, cwd=str(CODE), env=dict(os.environ, PYTHONPATH=f"{CODE}:{CODE / 'scripts'}"))
        except Exception as exc:
            log("training_retry", attempt=attempt, error=str(exc), resume=str(resume) if resume.is_dir() else None)
            time.sleep(min(300, 15 * attempt)); continue
        if (run_root / "DONE.json").is_file(): log("training_done", run=str(run_root)); return
    raise RuntimeError("K7-B exceeded recovery retry budget")


def main():
    log("orchestrator_start", arm="K7-B", corrected_from="K7-A", run=RUN_ID, v3=str(V3), source_commit=COMMIT)
    wait_parts(); extract_source(); meta = metadata(); build_dataset(meta); build_caches(); preflight(); train()
    log("orchestrator_complete", run=RUN_ID)


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        log("orchestrator_failed", error=repr(exc)); raise
