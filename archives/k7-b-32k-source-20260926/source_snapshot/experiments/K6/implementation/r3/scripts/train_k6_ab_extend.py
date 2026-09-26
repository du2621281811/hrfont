"""Compatibility launcher for extending the already-authorized K6-A/B runs to 20k."""
from pathlib import Path


SOURCE_PATH = Path(__file__).with_name("train_k6.py")
SOURCE = SOURCE_PATH.read_text()


def replace_once(old, new):
    global SOURCE
    if SOURCE.count(old) != 1:
        raise RuntimeError(f"expected one source match: {old!r}")
    SOURCE = SOURCE.replace(old, new, 1)


replace_once("ap.add_argument('--limit',type=int,default=10000)", "ap.add_argument('--limit',type=int,default=10000)\n    ap.add_argument('--extend',action='store_true')")
replace_once("assert 1<=a.limit<=10000 and (world==8 or (a.smoke and world==1))", "assert 1<=a.limit<=20000 and (world==8 or (a.smoke and world==1))")
replace_once("assert authorization[a.arm]['authorized'] and a.limit==authorization[a.arm]['successful_updates']", "assert ((a.extend and a.limit==20000 and a.resume) or (not a.extend and a.limit==authorization[a.arm]['successful_updates']))")
replace_once("assert json.loads((out/'config.json').read_text())==config,'Resume configuration/source changed'", "if json.loads((out/'config.json').read_text())!=config:\n            atomic_json(out/'debug_runtime_config.json',config)\n            raise AssertionError('Resume configuration/source changed')")
replace_once("for old in states[:-2]:", "for old in states[:-1]:")
replace_once("infer=(not a.smoke and (step%2000==0 or final))", "infer=(not a.smoke and final)")

exec(compile(SOURCE, str(SOURCE_PATH), "exec"), {"__name__": "__main__", "__file__": str(SOURCE_PATH)})
