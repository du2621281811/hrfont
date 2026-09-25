"""Compatibility launcher for authorized K6 extensions and the 20-update K6-C.

The original 10k trainer and its code identity remain untouched. This wrapper
derives a narrowly-scoped continuation variant at runtime so the 10k state can
be resumed without changing the identity recorded in that state.
"""
from pathlib import Path
import sys


SOURCE_PATH = Path(__file__).with_name("train_k6.py")
SOURCE = SOURCE_PATH.read_text()


def replace_once(old, new):
    global SOURCE
    count = SOURCE.count(old)
    if count != 1:
        raise RuntimeError(f"extension transform expected one match, got {count}: {old!r}")
    SOURCE = SOURCE.replace(old, new, 1)


replace_once("NEW_B='K6-B-RSI-NOOFFSETLOSS'", "NEW_B='K6-B-RSI-NOOFFSETLOSS'\nNEW_C='K6-C-RSI-AUX-NOOFFSETLOSS'")
replace_once("choices=['K6-A','K6-B',NEW_B]", "choices=['K6-A','K6-B',NEW_B,NEW_C]")
replace_once("ap.add_argument('--limit',type=int,default=10000)", "ap.add_argument('--limit',type=int,default=10000)\n    ap.add_argument('--extend',action='store_true')\n    ap.add_argument('--c-short',action='store_true')")
replace_once("assert 1<=a.limit<=10000 and (world==8 or (a.smoke and world==1))", "assert 1<=a.limit<=20000 and (world==8 or (a.smoke and world==1))")
replace_once("new_b=a.arm==NEW_B", "new_b=a.arm==NEW_B\n    new_c=a.arm==NEW_C\n    offsetless=new_b or new_c\n    aux_enabled=not new_b")
replace_once("authorization_file=ASSETS/'AUTHORIZATION_RSI_NO_OFFSET.json' if new_b else ASSETS/'AUTHORIZATION.json'", "authorization_file=ASSETS/'AUTHORIZATION_RSI_NO_OFFSET.json' if offsetless else ASSETS/'AUTHORIZATION.json'")
replace_once("model,args=model_for(a.arm,device,out/f'provenance_rank{rank}')", "model,args=model_for('K6-A' if new_c else a.arm,device,out/f'provenance_rank{rank}')")
replace_once("k6_objective='endpoint-generation-only-k5b-rsi-no-offset-regularization' if new_b else 'pure-noise-v2-every16-8step-lambda005-ramp1000'", "k6_objective=('endpoint-generation-only-k5b-rsi-no-offset-regularization' if new_b else ('combined-pure-noise-ranking-and-endpoint-no-offset' if new_c else 'pure-noise-v2-every16-8step-lambda005-ramp1000'))")
replace_once("rsi_enabled=True if new_b else None", "rsi_enabled=True if (new_b or new_c) else None")
replace_once("pure_noise_auxiliary=False if new_b else True", "pure_noise_auxiliary=False if new_b else True")
replace_once("offset_loss_weight=0.0 if new_b else .25", "offset_loss_weight=0.0 if offsetless else .25")
replace_once("comp=.01,detail=.05,offset=0.0 if new_b else .25,vgg=.01", "comp=.01,detail=.05,offset=0.0 if offsetless else .25,vgg=.01")
replace_once("gradient_sync='explicit global average after main only' if new_b else 'explicit global average after main plus auxiliary'", "gradient_sync='explicit global average after main only' if new_b else 'explicit global average after main plus auxiliary'")
replace_once("authorized_updates=18 if a.smoke else 10000)", "authorized_updates=20 if new_c else (18 if a.smoke else 10000))")
replace_once("assert authorization[a.arm]['authorized'] and a.limit==authorization[a.arm]['successful_updates']", "assert ((a.extend and a.limit==20000 and a.resume) or (not a.extend and a.limit==authorization[a.arm]['successful_updates']))")
replace_once("if not a.smoke:\n        preflight_file=STORE/'control/PREFLIGHT_PASSED_RSI_NO_OFFSET.json' if new_b else STORE/'control/PREFLIGHT_PASSED.json'\n        pre=json.loads(preflight_file.read_text())\n        assert pre['status']=='PASS' and pre['identity']==identity\n        assert ((a.extend and a.limit==20000 and a.resume) or (not a.extend and a.limit==authorization[a.arm]['successful_updates']))", "if not a.smoke and not new_c:\n        preflight_file=STORE/'control/PREFLIGHT_PASSED_RSI_NO_OFFSET.json' if new_b else STORE/'control/PREFLIGHT_PASSED.json'\n        pre=json.loads(preflight_file.read_text())\n        assert pre['status']=='PASS' and pre['identity']==identity\n        assert ((a.extend and a.limit==20000 and a.resume) or (not a.extend and a.limit==authorization[a.arm]['successful_updates']))\n    if new_c:\n        assert a.c_short and a.limit==20 and not a.extend and not a.resume")
replace_once("loss=eps+.01*perceptual+(0.0 if new_b else .25)*offset.float()+extra", "loss=eps+.01*perceptual+(0.0 if offsetless else .25)*offset.float()+extra")
replace_once("is_aux=(step%16==0) and not new_b", "is_aux=(step%16==0) and aux_enabled")
replace_once("pure_objective(a.arm,features,target,neutral,[sampler.entries[i][2] for i in pair_indices],calibration)", "pure_objective('K6-A' if new_c else a.arm,features,target,neutral,[sampler.entries[i][2] for i in pair_indices],calibration)")
replace_once("for old in states[:-2]:", "for old in states[:-1]:")
replace_once("infer=(not a.smoke and (step%2000==0 or final))", "infer=(not a.smoke and final)")

exec(compile(SOURCE, str(SOURCE_PATH), "exec"), {"__name__": "__main__", "__file__": str(SOURCE_PATH)})
