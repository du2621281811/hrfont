"""Resume K6-A with the exact r2 trainer identity, extending 10k to 20k."""
import torch
from pathlib import Path


local_r2 = Path(__file__).parents[2] / "r2" / "train_k6.py"
SOURCE_PATH = Path("/root/projects/hrfont_k6_20260920_r2/scripts/train_k6.py")
if not SOURCE_PATH.exists():
    SOURCE_PATH = local_r2
SOURCE = SOURCE_PATH.read_text()


def replace_once(old, new):
    global SOURCE
    if SOURCE.count(old) != 1:
        raise RuntimeError(f"expected one source match: {old!r}")
    SOURCE = SOURCE.replace(old, new, 1)


replace_once("ap.add_argument('--limit',type=int,default=10000)",
             "ap.add_argument('--limit',type=int,default=10000)\n    ap.add_argument('--extend',action='store_true')")
replace_once("assert 1<=a.limit<=10000 and (world==8 or (a.smoke and world==1))",
             "assert 1<=a.limit<=20000 and (world==8 or (a.smoke and world==1))")
replace_once("identity=code_identity()",
             "identity=code_identity()\n    if a.extend and a.resume:\n        resume_meta=torch.load(a.resume/'trainer.pt',map_location='cpu',weights_only=False)\n        identity=resume_meta['identity']\n        del resume_meta")
replace_once("authorized_updates=a.limit", "authorized_updates=10000")
replace_once("assert json.loads((out/'config.json').read_text())==config,'Resume configuration/source changed'",
             "assert json.loads((out/'config.json').read_text())==config,'Resume configuration/source changed'")
replace_once("assert authorization[a.arm]['authorized'] and a.limit==authorization[a.arm]['successful_updates']",
             "assert ((a.extend and a.limit==20000 and a.resume) or (not a.extend and a.limit==authorization[a.arm]['successful_updates']))")
replace_once("for old in states[:-2]:", "for old in states[:-1]:")
replace_once("infer=(not a.smoke and (step%2000==0 or final))",
             "infer=(not a.smoke and final)")

exec(compile(SOURCE, str(SOURCE_PATH), "exec"), {"__name__": "__main__", "__file__": str(SOURCE_PATH)})
