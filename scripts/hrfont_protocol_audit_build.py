#!/usr/bin/env python3
"""Build PROTOCOL_AUDIT.json — per-experiment input rendering/processing + version binding.

Ground truth = metrics JSON timestamps + checkpoint mtimes + current script fingerprints.
Flags when eval scripts were modified AFTER a result was recorded.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight"
OUT = REP / "formal_preview"
FE = REP / "formal_eval"
ABL = REP / "dropout_ablation"

# Shared preprocessing (matches torchvision in train/eval loaders)
TFM = "Resize(96) → ToTensor → Normalize(mean=0.5, std=0.5)"


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _file_info(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "exists": False}
    st = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        "sha256_16": _sha256(path),
        "bytes": st.st_size,
    }


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _script_binding(name: str, rel: str, eval_t: str | None) -> dict:
    p = ROOT / rel
    info = _file_info(p)
    newer = False
    if eval_t and info.get("mtime"):
        newer = info["mtime"] > eval_t
    return {
        "name": name,
        **info,
        "eval_recorded_at": eval_t,
        "script_newer_than_eval": newer,
        "note": (
            "⚠️ 当前脚本晚于该次评测；主表数字以 metrics JSON 为准，勿用现行脚本反推当时配置"
            if newer
            else "脚本时间不晚于评测记录"
        ),
    }


def _input(
    *,
    role: str,
    source: str,
    render: str,
    script: str,
    size: str,
    preprocess: str,
    train_eval: str,
    detail: str = "",
) -> dict:
    return {
        "role": role,
        "source": source,
        "render": render,
        "script": script,
        "size": size,
        "preprocess": preprocess,
        "train_vs_eval": train_eval,
        "detail": detail,
    }


def build_audit() -> dict:
    m80 = _load_json(FE / "metrics_step_80000.json")
    mb = _load_json(FE / "metrics_stageB_25000.json")
    viz_meta = _load_json(OUT / "viz_meta.json")

    t80 = m80.get("t")
    tB = mb.get("t")
    ck_ft = ROOT / "runs/ft_cnstyle/global_step_25000"
    ck_a = ROOT / "runs/e2_stageA96_formal/step_80000.pt"
    ck_b = ROOT / "runs/e2_stageB96/step_25000.pt"

    common_eval_inputs = [
        _input(
            role="Content",
            source="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf + 字符 c",
            render="binary-search 最大字号 + 居中（14 iter, margin 8px）",
            script="scripts/hrfont_e2_official_eval.render",
            size="96×96 RGB",
            preprocess=TFM,
            train_eval="❌ 训练用预渲 ContentImage（见下）",
            detail="P1 全字符；seed=123",
        ),
        _input(
            role="Style",
            source="Demo-8 测试字体 TTF +「永」",
            render="同上 render()",
            script="scripts/hrfont_e2_official_eval.render",
            size="96×96 RGB",
            preprocess=TFM,
            train_eval="❌ 训练 style 来源不同（见各实验 train）",
            detail="1-shot 固定「永」；STYLE_CHAR 来自 data/retrain_v2/meta.json",
        ),
        _input(
            role="GT（L1）",
            source="data/unified_v1/renders/128/gt_latin/<font>/<char>.png",
            render="build_retrain_v2_dataset @128 L 灰度",
            script="scripts/build_retrain_v2_dataset.render",
            size="128→96 bilinear（to_arr）",
            preprocess="灰度 /255",
            train_eval="— 仅评测",
            detail="bank r96 fallback 若缺失",
        ),
    ]

    experiments = {
        "ft_cnstyle@25k": {
            "question": "官方 RSI 跨语微调基线",
            "train_script": "scripts/retrain_v2_finetune_fontdiffuser.py → code/FontDiffuser/train.py",
            "eval_script": "scripts/hrfont_e2_official_eval.py (FontDiffuserOfficialDPM)",
            "ckpt": _file_info(ck_ft / "unet.pth"),
            "metrics": {
                "L1_mean": m80.get("ft_cnstyle", {}).get("L1_mean"),
                "n": m80.get("ft_cnstyle", {}).get("n"),
                "t": t80,
                "json": "reports/hrfont_overnight/formal_eval/metrics_step_80000.json",
            },
            "train_inputs": [
                _input(
                    role="Content",
                    source="data/fontdiffuser/train/ContentImage/uXXXX.jpg",
                    render="build_retrain_v2_dataset: DejaVu binary-search @128 灰度 → jpg RGB",
                    script="scripts/build_retrain_v2_dataset.py + build_retrain_v2_fontdiffuser_data.py",
                    size="128 源图 → FontDataset Resize(96)",
                    preprocess=TFM,
                    train_eval="❌ eval 为 live @96 RGB（算法同，分辨率/色彩空间不同）",
                ),
                _input(
                    role="Style",
                    source="data/.../StyleImage/<font>/<font>+uXXXX.jpg",
                    render="同字体中文池全量 PNG→jpg（非仅 ref8）",
                    script="scripts/build_retrain_v2_fontdiffuser_data.py",
                    size="128→96",
                    preprocess=TFM,
                    train_eval="❌ eval 固定 live「永」",
                    detail="FontDataset 随机采样同字体任一汉字",
                ),
                _input(
                    role="Target",
                    source="data/.../TargetImage/<font>/<font>+uXXXX.jpg",
                    render="训练字体英文 GT @128 灰度→jpg",
                    script="scripts/build_retrain_v2_fontdiffuser_data.py",
                    size="128→96",
                    preprocess=TFM,
                    train_eval="—",
                ),
                _input(
                    role="RSI 结构分支",
                    source="Ec(style_image)",
                    render="content_encoder(style) 多尺度残差",
                    script="code/FontDiffuser/src/model.py",
                    size="特征空间",
                    preprocess="—",
                    train_eval="✅ eval 同 Ec(style)",
                    detail="官方 FontDiffuser RSI；无 Δ",
                ),
                _input(
                    role="Δ / Support",
                    source="—",
                    render="—",
                    script="—",
                    size="—",
                    preprocess="—",
                    train_eval="—",
                ),
            ],
            "eval_inputs": common_eval_inputs
            + [
                _input(
                    role="RSI 结构分支",
                    source="Ec(live style「永」)",
                    render="FontDiffuserOfficialDPM.forward",
                    script="scripts/hrfont_e2_official_eval.py",
                    size="特征空间",
                    preprocess="—",
                    train_eval="⚠️ 与 train 同公式，style 图来源不同",
                ),
            ],
        },
        "stageA_formal@80k": {
            "question": "RSI 改接 Δ = Ec(α混库) − Ec(content)",
            "train_script": "scripts/hrfont_e2_stageA96_formal_train.py",
            "eval_script": "scripts/hrfont_e2_official_eval.py (FontDiffuserDeltaDPM)",
            "ckpt": _file_info(ck_a),
            "metrics": {
                "L1_mean": m80.get("stageA_formal", {}).get("L1_mean"),
                "n": m80.get("stageA_formal", {}).get("n"),
                "t": t80,
                "step": 80000,
                "json": "reports/hrfont_overnight/formal_eval/metrics_step_80000.json",
            },
            "train_inputs": [
                _input(
                    role="Content",
                    source="ContentImage/uXXXX.jpg（与 ft 同源）",
                    render="预渲 DejaVu @128→96",
                    script="StageAFormalSet.content_cache",
                    size="96",
                    preprocess=TFM,
                    train_eval="❌ eval live DejaVu @96",
                ),
                _input(
                    role="Style",
                    source="data/hrfont/e0_bank/r96/<font>/uXXXX.png",
                    render="render_fit ~8% 边距",
                    script="scripts/hrfont_e0_build_bank.render_fit",
                    size="96 RGB",
                    preprocess=TFM,
                    train_eval="❌ eval live「永」",
                    detail="随机 ref8 之一；非 1-shot 固定",
                ),
                _input(
                    role="Target",
                    source="bank r96 同字体同字 PNG",
                    render="render_fit",
                    script="hrfont_e0_build_bank",
                    size="96",
                    preprocess=f"{TFM}; target_01 denorm for VGG",
                    train_eval="—",
                ),
                _input(
                    role="Δ",
                    source="bank r96 top-3 训练字体同字 PNG 像素加权混",
                    render="α=softmax(cos(proto)/τ) τ=0.07 M=3；像素 mix 后 Ec 作差",
                    script="StageAFormalSet._alpha + train forward",
                    size="96 RGB tensor",
                    preprocess=f"{TFM} on mix PNG",
                    train_eval="⚠️ eval 同 α 逻辑；bank 为 render_fit 非 live",
                    detail="dropout 25% zero delta_res",
                ),
                _input(
                    role="RSI 结构分支",
                    source="Ec(Δ_img) − Ec(content)",
                    render="delta_res = mix_res − content_res",
                    script="hrfont_e2_stageA96_formal_train.py",
                    size="特征差",
                    preprocess="—",
                    train_eval="✅ eval FontDiffuserDeltaDPM 同式",
                ),
            ],
            "eval_inputs": common_eval_inputs
            + [
                _input(
                    role="Δ",
                    source="alpha_mix(proto, bank, train, c)",
                    render="测试字体 proto 不在 cache 时用 ref8 现场 Ec 均值",
                    script="hrfont_e2_official_eval.alpha_mix",
                    size="96 RGB",
                    preprocess=TFM,
                    train_eval="⚠️ 像素混 bank PNG；与 train 同 α",
                ),
                _input(
                    role="RSI 结构分支",
                    source="Ec(Δ) − Ec(content)",
                    render="FontDiffuserDeltaDPM._hidden",
                    script="hrfont_e2_official_eval.py",
                    size="特征差",
                    preprocess="—",
                    train_eval="✅",
                ),
            ],
        },
        "stageB_support@25k": {
            "question": "冻 UNet，训 SupportAdapter + gap-gated support",
            "train_script": "scripts/hrfont_e2_stageB96_train.py",
            "eval_script": "scripts/hrfont_e2_stageB_official_eval.py",
            "ckpt": _file_info(ck_b),
            "metrics": {
                "L1_mean": mb.get("stageB", {}).get("L1_mean"),
                "n": mb.get("stageB", {}).get("n"),
                "n_with_support": mb.get("stageB", {}).get("n_with_support"),
                "t": tB,
                "step": 25000,
                "json": "reports/hrfont_overnight/formal_eval/metrics_stageB_25000.json",
            },
            "train_inputs": [
                _input(
                    role="Content/Style/Target/Δ",
                    source="同 Stage A formal",
                    render="同 StageAFormalSet / StageBSet",
                    script="hrfont_e2_stageB96_train.py StageBSet",
                    size="96",
                    preprocess=TFM,
                    train_eval="同 A 的 train 列",
                ),
                _input(
                    role="Support",
                    source="pick_support → bank r96 支撑字图",
                    render="gap≥0.35 → MMR top-3 q × α 字体混",
                    script="scripts/hrfont_support_utils.pick_support",
                    size="96 RGB",
                    preprocess=f"{TFM} → pool_ec → SupportAdapter",
                    train_eval="⚠️ eval 需 font_proto=proto（8/29 前 bug 致 support=0）",
                    detail="train 20% drop support",
                ),
            ],
            "eval_inputs": common_eval_inputs
            + [
                _input(
                    role="Δ",
                    source="同 Stage A eval",
                    render="alpha_mix + set_cond",
                    script="hrfont_e2_stageB_official_eval.py",
                    size="96",
                    preprocess=TFM,
                    train_eval="✅ 同 A",
                ),
                _input(
                    role="Support",
                    source="pick_support(..., font_proto=proto)",
                    render="SupportAdapter → concat style_hidden",
                    script="hrfont_e2_stageB_official_eval.py + hrfont_support_utils",
                    size="token",
                    preprocess="pool_ec",
                    train_eval="✅ 有效 run 952/976",
                ),
            ],
        },
        "dropout_ablation@10k": {
            "question": "仅改 dropout，10k 步趋势（非主表）",
            "train_script": "scripts/hrfont_e2_dropout_ablation_train.py",
            "eval_script": "scripts/hrfont_e2_official_eval.py --ckpt ablation --skip-ft",
            "metrics_source": "reports/hrfont_overnight/dropout_ablation/eval_*.json",
            "note": "勿与 R0@80k 比绝对 L1；ft 行来自 metrics_step_80000 缓存",
            "train_inputs": "同 stageA_formal@80k，仅 DROP_CFG/DROP_DELTA/drop_mode 不同",
            "eval_inputs": "同 stageA eval 协议",
        },
        "formal_viz": {
            "question": "GT|ft|A|B 可视化",
            "script": "scripts/hrfont_formal_viz_build.py",
            "metrics": viz_meta,
            "eval_inputs": "与 official eval 相同（DejaVu live、1-shot永、DPM20、CFG7.5）",
            "chars": "AaR8 子集（32 格）非全 P1",
        },
        "e3_rsi_style@10k": {
            "question": "对照：同 formal 数据/步数，RSI 仍接 Ec(style) 而非 Δ",
            "train_script": "scripts/hrfont_e3_rsi_style_train.py",
            "eval_script": "scripts/hrfont_e2_official_eval.py --rsi-style",
            "ckpt": _file_info(ROOT / "runs/e3_rsi_style_control/step_10000.pt"),
            "metrics_json": "reports/hrfont_overnight/formal_eval/metrics_e3_10000.json",
            "train_inputs": "同 Stage A（Content/Style/Target bank）；RSI=Ec(style) 官方接线",
            "eval_inputs": "同 ft official（无 Δ）；FontDiffuserOfficialDPM",
            "status": "pending until metrics_e3_10000.json exists",
        },
    }

    invalid_results = [
        {
            "t": "2026-08-29 01:10:04",
            "experiment": "stageB@25k",
            "L1": 0.0879,
            "n_with_support": 0,
            "reason": "pick_support 未传 font_proto；测试字体不在 style_proto cache",
            "superseded_by": "metrics_stageB_25000.json @ 2026-08-29 02:07:08",
            "log": "reports/hrfont_overnight/eval_stageB.log",
        },
    ]

    version_bindings = [
        _script_binding("official_eval", "scripts/hrfont_e2_official_eval.py", t80),
        _script_binding("stageB_eval", "scripts/hrfont_e2_stageB_official_eval.py", tB),
        _script_binding("support_utils", "scripts/hrfont_support_utils.py", tB),
        _script_binding("formal_viz", "scripts/hrfont_formal_viz_build.py", viz_meta.get("t")),
        _script_binding("stageA_train", "scripts/hrfont_e2_stageA96_formal_train.py", "2026-08-26 01:10:28"),
        _script_binding("stageB_train", "scripts/hrfont_e2_stageB96_train.py", "2026-08-28 19:58:00"),
    ]

    train_eval_gaps = [
        {
            "item": "Content",
            "train": "ContentImage 预渲 DejaVu @128 灰度→jpg→Resize96",
            "eval": "live DejaVu binary-search @96 RGB",
            "impact": "domain gap；ft/A/B 训练一致、eval 一致，互相公平",
        },
        {
            "item": "Style",
            "train": "A/B: bank render_fit 随机 ref8；ft: 随机 CN 池",
            "eval": "live 测试字体「永」",
            "impact": "train/eval style 分布不同；三方法 eval 相同",
        },
        {
            "item": "Bank / Δ 混图",
            "train": "render_fit ~8% 边距",
            "eval": "同 bank PNG；非 live 渲染",
            "impact": "Δ 输入与 live content/style 不一致（设计如此）",
        },
        {
            "item": "Target/GT",
            "train": "bank 或 TargetImage @128",
            "eval": "gt_latin @128→96",
            "impact": "L1 对齐 unified_v1 GT",
        },
    ]

    any_newer = any(v.get("script_newer_than_eval") for v in version_bindings if v.get("eval_recorded_at"))

    return {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "principle": "主表 L1 以 metrics JSON 时间戳为准；本文件描述绑定到当时 ckpt + 日志；现行脚本若晚于评测仅用于复现说明",
        "script_drift_warning": any_newer,
        "experiments": experiments,
        "train_eval_gaps": train_eval_gaps,
        "version_bindings": version_bindings,
        "invalid_results": invalid_results,
        "checkpoints": {
            "ft@25k": _file_info(ck_ft / "unet.pth"),
            "stageA@80k": _file_info(ck_a),
            "stageB@25k": _file_info(ck_b),
        },
    }


def main() -> None:
    audit = build_audit()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "PROTOCOL_AUDIT.json"
    out_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path} drift_warning={audit['script_drift_warning']}")


if __name__ == "__main__":
    main()
