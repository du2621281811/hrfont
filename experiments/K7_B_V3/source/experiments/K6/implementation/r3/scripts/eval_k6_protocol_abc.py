"""K6 A/B/C normal-protocol evaluator; C reuses the validated RSI runtime."""
from pathlib import Path


SOURCE_PATH = Path(__file__).with_name("eval_k6_protocol.py")
SOURCE = SOURCE_PATH.read_text()


def replace_once(old, new):
    global SOURCE
    if SOURCE.count(old) != 1:
        raise RuntimeError(f"expected one source match: {old!r}")
    SOURCE = SOURCE.replace(old, new, 1)


C = "K6-C-RSI-AUX-NOOFFSETLOSS"
replace_once('parser.add_argument("--arm", choices=["K6-A", "K6-B", "K6-B-RSI-NOOFFSETLOSS"], required=True)',
             'parser.add_argument("--arm", choices=["K6-A", "K6-B", "K6-B-RSI-NOOFFSETLOSS", "K6-C-RSI-AUX-NOOFFSETLOSS"], required=True)')
replace_once('model, runtime_args = model_for(args.arm, device, args.out / f"provenance_rank{rank}", evaluation=True)',
             'model, runtime_args = model_for("K6-B-RSI-NOOFFSETLOSS" if args.arm == "K6-C-RSI-AUX-NOOFFSETLOSS" else args.arm, device, args.out / f"provenance_rank{rank}", evaluation=True)\n    if args.arm == "K6-C-RSI-AUX-NOOFFSETLOSS": model.arm = args.arm')

exec(compile(SOURCE, str(SOURCE_PATH), "exec"), {"__name__": "__main__", "__file__": str(SOURCE_PATH)})
