#!/usr/bin/env bash
# Build migrate blobs for a second host.
# Protocol A (166k PNGs) and Ec (~94GB) are rsync'd, not tarred.
set -euo pipefail
ROOT=/root/projects/hrfont
OUT="${1:-$ROOT/artifacts/migrate_v100}"
mkdir -p "$OUT"
cd "$ROOT"

need() {
  if [[ ! -e "$1" ]]; then
    echo "missing: $ROOT/$1" >&2
    exit 2
  fi
}

need data/fontdiffuser-p253-t295-s338-cn2west-v2
need artifacts/f0/es_spatial_f0
need artifacts/f0/ec_multiscale_f0/manifest.json
need artifacts/f0/support_bank.json
need runs/F0-RSIFREE-FT-A-S3407/best/unet.pth
need code/official/FontDiffuser/ckpt/unet.pth

tar_h() { tar -C "$ROOT" -h -cf "$@"; }
tar_plain() { tar -C "$ROOT" -cf "$@"; }

echo "==> $OUT"
{
  date -Is
  hostname
  git -C "$ROOT" rev-parse --short HEAD
} | tee "$OUT/BUILT_AT.txt"

if [[ ! -s "$OUT/01_small.tar" ]]; then
  echo "==> 01_small (support banks + official P1 ckpt)"
  tar_h "$OUT/01_small.tar" \
    artifacts/f0/support_bank.json \
    artifacts/f0/support_bank_f3b_stroke.json \
    artifacts/f0/support_bank_f3b_topology.json \
    artifacts/f0/support_bank_f3b_topology.manifest.json \
    artifacts/f0/topology_signatures_f3b.json \
    artifacts/f0/es_cosine_f0.f16.json \
    code/official/FontDiffuser/ckpt/unet.pth \
    code/official/FontDiffuser/ckpt/style_encoder.pth \
    code/official/FontDiffuser/ckpt/content_encoder.pth \
    code/official/FontDiffuser/ckpt/scr_210000.pth \
    reports/artifacts_sync
else
  echo "==> skip 01_small (exists)"
fi

echo "==> 03_es_cache (~1.7GB, few files)"
tar_plain "$OUT/03_es_cache.tar" artifacts/f0/es_spatial_f0

echo "==> 04_f0_best (~1.1GB, follow best symlink)"
tar_h "$OUT/04_f0_best.tar" \
  runs/F0-RSIFREE-FT-A-S3407/best \
  runs/F0-RSIFREE-FT-A-S3407/DONE.json \
  runs/F0-RSIFREE-FT-A-S3407/F0-RSIFREE-FT-A-S3407_config.yaml \
  runs/F0-RSIFREE-FT-A-S3407/launch_meta.json

echo "==> 05_eval_ckpts (F1@80k F2@75k+80k F3@80k)"
tar_plain "$OUT/05_eval_ckpts.tar" \
  runs/F1-OFFRSI-A-S3407/global_step_80000 \
  runs/F1-OFFRSI-A-S3407/DONE.json \
  runs/F1-OFFRSI-A-S3407/launch_meta.json \
  runs/F1-OFFRSI-A-S3407/F1-OFFRSI-A-S3407_config.yaml \
  runs/F2-DELTARSI-A-S3407/global_step_75000 \
  runs/F2-DELTARSI-A-S3407/global_step_80000 \
  runs/F2-DELTARSI-A-S3407/DONE.json \
  runs/F2-DELTARSI-A-S3407/launch_meta.json \
  runs/F2-DELTARSI-A-S3407/F2-DELTARSI-A-S3407_config.yaml \
  runs/F3-JOINT-DS-A-S3407/global_step_80000 \
  runs/F3-JOINT-DS-A-S3407/DONE.json \
  runs/F3-JOINT-DS-A-S3407/launch_meta.json \
  runs/F3-JOINT-DS-A-S3407/F3-JOINT-DS-A-S3407_config.yaml

{
  echo "# HR-Font migrate pack  (gitignored; not for GitHub)"
  echo "# Built $(date -Is)  HEAD=$(git -C "$ROOT" rev-parse --short HEAD)"
  echo
  echo "## Tarballs (unpack at /root/projects/hrfont after git clone)"
  ls -lh "$OUT"/*.tar | awk '{print $5, $9}'
  echo
  echo "## rsync these (not tarred)"
  echo "  data/fontdiffuser-p253-t295-s338-cn2west-v2/   # 0.7GB, 166k PNGs"
  echo "  artifacts/f0/ec_multiscale_f0/                 # 94GB"
  echo
  echo "  bash scripts/rsync_newhost_migrate.sh user@NEWHOST"
  echo
  echo "## Unpack tars"
  echo "  bash scripts/unpack_newhost_migrate.sh /root/projects/hrfont/artifacts/migrate_v100"
  echo
  echo "## Do not copy"
  echo "  whole runs/  (F0 23G, F1/F2/F3 ~19G each)"
  echo "  data/fontdiffuser data/font data/hrfont_bank"
  echo "  artifacts/e12  (unless doing E12)"
} | tee "$OUT/MANIFEST.txt"

echo "OK pack -> $OUT"
ls -lh "$OUT"
