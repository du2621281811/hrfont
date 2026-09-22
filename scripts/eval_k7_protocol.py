"""K7-B wrapper around the frozen K6 protocol evaluator.

The runtime already supports the K7-B arm; the upstream evaluator's argparse
choices still list only the original K6 arm names.
"""
from pathlib import Path


SOURCE_PATH = Path("/root/projects/hrfont_k7_v3_20260922/experiments/K6/implementation/r3/scripts/eval_k6_protocol.py")
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
old = 'parser.add_argument("--arm", choices=["K6-A", "K6-B", "K6-B-RSI-NOOFFSETLOSS"], required=True)'
new = 'parser.add_argument("--arm", choices=["K6-A", "K6-B", "K6-B-RSI-NOOFFSETLOSS", "K7-B"], required=True)'
if SOURCE.count(old) != 1:
    raise RuntimeError("K7 evaluator argparse contract changed")
SOURCE = SOURCE.replace(old, new, 1)
exec(compile(SOURCE, str(SOURCE_PATH), "exec"), {"__name__": "__main__", "__file__": str(SOURCE_PATH)})
