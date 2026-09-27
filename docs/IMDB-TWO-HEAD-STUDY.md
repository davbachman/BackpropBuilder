# Two-head IMDb transformer comparison

Three new runs (seeds 137, 211, 307) compared with the existing three one-head controls. Exactly the same 20,000 training reviews, 1,000 validation reviews, training-fitted 4,000-entry vocabulary, context 128, total model width 32 and 140,985 parameters. No final-test evaluation.

Two heads each have width 16 and independent query/key/value projections, scaled scores (divide by sqrt(16)), and softmax. Their outputs concatenate to width 32 before the unchanged output projection. Initial projection matrices are column partitions of the one-head matrices. Every other initial weight is unchanged. This avoids increasing model capacity by doubling its width. All components remain editable elementary blocks; no prebuilt attention operator is introduced.

AdamW learning rate 0.001, no dropout, batch 32, weight decay 0.01, global gradient clipping 1, up to 30 epochs, patience 3, minimum validation-loss improvement 0.001. Each run restores its selected validation-loss checkpoint. Training uses the native PyTorch graph interpreter. Reported times are native CPU training measurements, not browser timings, and controls were run earlier.

| Heads | Mean validation loss | Mean accuracy | Accuracy range | Selected epochs | Mean seconds |
|---:|---:|---:|---:|---|---:|
| 1 | 0.4052 | 81.2% | 80.4%–81.7% | 2, 2, 2 | 10.7 |
| 2 | 0.4026 | 81.5% | 81.0%–81.9% | 2, 2, 2 | 15.4 |

Accuracy change: 0.30 percentage points. Validation-loss change: -0.0027 (negative is better). Ranges describe seed variation, not confidence intervals. These are exploratory results on repeatedly used validation reviews.

Import `output/imdb-two-head/models/transformer-two-heads.json` to inspect the two-head checkpoint (fixed seed 137, accuracy 81.0%). Initial parameters are included for Reset. Set batch 32 and up to 30 epochs explicitly; stored optimizer settings do not include optimizer moments. All generated datasets and models stay git-ignored. The one-head control is `output/imdb-data-vocab/models/transformer-n20000-v4000.json`.

Reproduce after the data/vocabulary study: node scripts/prepare-two-head-study.mjs; python3 scripts/train-data-vocab-study.py --root output/imdb-two-head; node scripts/report-two-head-study.mjs. Python requires PyTorch. Preparation accepts DATA_VOCAB_OUTPUT and TWO_HEAD_OUTPUT; reporting accepts TWO_HEAD_OUTPUT and reads the saved baseline path from protocol.json.

The 0.3-point mean accuracy increase is small relative to variation across three initializations. Two heads are a valid working variant, but this experiment does not establish a reliable accuracy advantage. All runs selected epoch 2. Final-test reviews remain unused.

Implementation: `withTwoAttentionHeads` in `src/test/textModels.ts` constructs two branches from existing app operations. The native study interpreter now supports feature concatenation. Tests check preserved parameters, trace/tensor losses and gradients on unequal review lengths, padding independence, and native PyTorch parity through three AdamW updates. The parity test compares gradients by parameter ID rather than assuming both engines visit nodes in the same order. A real WebGL update at width 32 and vocabulary 4,000 produced finite weights and matched trace loss within 3e-8. Seven targeted tests, lint, and the production build passed.

The [final context and matched-model comparison](IMDB-FINAL-COMPARISON.md) selects a longer context on validation and then evaluates frozen checkpoints on a fresh test sample.
