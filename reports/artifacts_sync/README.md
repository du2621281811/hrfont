# Syncable small artifacts (git-safe)

Copies of files that live under gitignored `artifacts/f0/` but are needed to reproduce F3 / F3b.

| File | Use |
|------|-----|
| `support_bank_f3_legacy.json` | Old F3 fixed ref8 own-font support table |
| `support_bank_f3b_stroke.json` | F3b preset CN + stroke-bucket pool |

Training launchers still read `artifacts/f0/...`. On a fresh clone, copy these back:

```bash
mkdir -p artifacts/f0
cp reports/artifacts_sync/support_bank_f3_legacy.json artifacts/f0/support_bank.json
cp reports/artifacts_sync/support_bank_f3b_stroke.json artifacts/f0/support_bank_f3b_stroke.json
```
