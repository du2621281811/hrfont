#!/usr/bin/env bash
# Unpack migrate_v100 tarballs into /root/projects/hrfont (git clone first).
set -euo pipefail
ROOT="${HRFONT_ROOT:-/root/projects/hrfont}"
PACK="${1:-$ROOT/artifacts/migrate_v100}"
mkdir -p "$ROOT"
cd "$ROOT"

if [[ ! -d "$ROOT/.git" ]]; then
  echo "clone the repo into $ROOT first:" >&2
  echo "  git clone git@github.com:du2621281811/hrfont.git $ROOT" >&2
  exit 2
fi

for t in 01_small 03_es_cache 04_f0_best 05_eval_ckpts; do
  f="$PACK/${t}.tar"
  if [[ -f "$f" ]]; then
    echo "unpack $f"
    tar -xf "$f" -C "$ROOT"
  else
    echo "skip missing $f"
  fi
done

# Launchers expect .../best
if [[ -d "$ROOT/runs/F0-RSIFREE-FT-A-S3407/best" ]]; then
  echo "F0 best ok"
fi
for pair in \
  "F1-OFFRSI-A-S3407:global_step_80000" \
  "F2-DELTARSI-A-S3407:global_step_80000" \
  "F3-JOINT-DS-A-S3407:global_step_80000"
 do
  run="${pair%%:*}"
  step="${pair##*:}"
  d="$ROOT/runs/$run"
  if [[ -d "$d/$step" && ! -e "$d/best" ]]; then
    ln -sfn "$step" "$d/best"
    echo "symlink $run/best -> $step"
  fi
done

echo "still needed if empty:"
echo "  data/fontdiffuser-p253-t295-s338-cn2west-v2/   (rsync, 0.7GB)"
echo "  artifacts/f0/ec_multiscale_f0/                 (rsync, 94GB)"
echo "then: conda env boogu (Python 3.10, torch 2.7.1+cu126, fp16, batch 8)"
echo "read docs/SETUP_COLLABORATOR.md"
