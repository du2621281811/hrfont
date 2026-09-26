# K7-B V3 implementation source archive

Download `K7_B_V3_code_review_20260926.zip` for the source, configuration/provenance, and data-contract snapshot intended for collaborator review. `SHA256SUMS` contains the archive checksum.

The archive omits evaluation image galleries, raw font-image datasets, model and encoder weights, teacher tensors, caches, checkpoints, logs, and predictions. After extracting it, read `K7_B_V3/README.md` and run `python3 K7_B_V3/verify_source_identity.py` to audit the 438 manifest-listed source files.

Important boundary: this is the V100 source snapshot captured on 2026-09-25, not a guarantee of byte-identical training-time source. The frozen configuration has different recorded hashes for `train_k6.py` and `k5_runtime.py`; the archive documents this explicitly. The current V100 host could not be reached for a fresh read-only check during packaging.
