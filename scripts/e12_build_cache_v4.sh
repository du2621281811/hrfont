#!/usr/bin/env bash
# Render cache_v4 from fonts_s3 using the v3 character axis.
# Does not touch artifacts/e12/external_font_cache or GPU2.
set -euo pipefail
PY=/root/miniforge3/envs/boogu/bin/python
ROOT=/root/projects/hrfont
FONTS=$ROOT/artifacts/e12/fonts_s3
CACHE=$ROOT/artifacts/e12/cache_v4
# 104 CN (v3) + 52 Latin + 10 digits
CHARS='永和书风骨韵天地一二三四五六七八九十人大小上下中山水火木金土日月年时分口目手足心生学国家民工力田车马鸟鱼虫石花草树林春夏秋冬东西南北左右前后来去出入开关高低长短多少新旧好坏黑白红黄蓝绿雨雪云电光明暗江海河湖原ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
test -d "$FONTS"
n=$(find "$FONTS" -type f \( -iname '*.ttf' -o -iname '*.otf' -o -iname '*.ttc' \) | wc -l)
echo "fonts_s3 files=$n"
if [[ "$n" -lt 25 ]]; then
  echo "need >=25 typefaces in $FONTS" >&2
  exit 1
fi
cd "$ROOT/scripts/eval_framework"
$PY build_cache.py \
  --fonts_dir "$FONTS" \
  --cache_dir "$CACHE" \
  --chars "$CHARS" \
  --strategy per_font_height_fit \
  --canvas 96 \
  --margin 6
