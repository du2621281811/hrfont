# E12 paper validation protocol (teacher / cosine)

**Primary:** Cross-Script Style Evaluator = shared encoder φ + mean-pooled Chinese refs + **cosine** Style Score.  
**Not** the main story: membership binary classifier (appendix only).

- Scorer: `e12_phi_s2_b_s3407` · `cosine_protocol.md`
- Data: `artifacts/e12/cache_v0913_b` (v0913_clean)
- Seed: 3407 · ref8: 永和书风骨韵天地
- Framing: **Cross-script / Style-aware / GT-free**

## Teacher validation tables (main deliverable)

See **`PAPER_TABLES_TEACHER.md`** and numbers in **`teacher_tables_1to3.json`**.

| Table | Question | Status |
|------:|----------|--------|
| 1 Retrieval | Does it retrieve correctly? | Done (+ CLIP/DINOv2/LPIPS) |
| 2 Verification | Does it discriminate correctly? | Done (+ baselines) |
| 3 Separation | Is score ordering sensible? | Done (appendix) |
| 4 Human | Does it agree with humans? | Board ready; ratings pending |

## Reproduce baselines

```bash
export HF_ENDPOINT=https://hf-mirror.com
/root/miniforge3/envs/boogu/bin/python scripts/e12_teacher_tables_baselines.py --device cuda:0
```
