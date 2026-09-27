# Final matched IMDb comparison

## Context selection (validation only)

| Context tokens | Mean validation accuracy | Mean validation loss |
|---:|---:|---:|
| 128 | 81.5% | 0.4026 |
| 256 | 85.7% | 0.3455 |

Selected context: 256, by mean validation loss across seeds 137, 211 and 307. The same 20,000 training / 1,000 validation reviews and training-fitted vocabulary of 4,000 were used. The 256-token run preserves the 128-token prefixes and shared initial weights, adding 128 position-embedding rows. Transformer width 32, two heads of width 16; no dropout.

## Frozen comparison on fresh test reviews

| Model | Mean validation accuracy | Mean test accuracy | Mean test loss | Test accuracy range | Selected epochs |
|---|---:|---:|---:|---|---|
| Linear word counts | 85.4% | 86.5% | 0.3227 | 86.4%–86.5% | 6, 4, 6 |
| MLP on word counts | 83.7% | 85.5% | 0.3317 | 85.5%–85.6% | 1, 1, 1 |
| Pooled learned embeddings | 85.6% | 86.3% | 0.3167 | 86.0%–86.6% | 3, 3, 3 |
| Two-head transformer | 85.7% | 85.2% | 0.3457 | 85.0%–85.5% | 2, 3, 3 |

Every figure averages the same three initializations; ranges are across seeds, not confidence intervals. Test accuracy uses 5,000 fresh, balanced official-test reviews, excluding the 1,000 reviews evaluated in the earlier accelerated study and the 200 pilot reviews. All twelve checkpoints and their hashes were frozen before preparing or evaluating this sample. Test results did not select architecture, context, epochs or a preferred seed. See frozen.json, holdout-manifest.json and test-results.json for the audit trail and per-review probabilities.

All four models use identical review IDs, vocabulary, and the same first 256 tokens. Word-count features are raw counts of that prefix, including unknown tokens. The MLP on counts has 12 hidden ReLU units. Pooled embeddings have width 32 and the same 12-unit classifier as the transformer, but no positions or attention. These controls separate nonlinear classification and learned embeddings from contextual processing, although architecture parameter counts differ.

Training: native PyTorch CPU using the app's graph primitives; AdamW learning rate 0.001, batch 32, decay 0.01, global norm clipping 1, no dropout, maximum 30 epochs, patience 3 and minimum validation-loss improvement 0.001. Validation loss selects each checkpoint. The optimizer and stopping budget are matched, not individually tuned to each architecture. This comparison does not establish a universal model ranking or isolate word order as the cause of a performance difference. Repeated prior validation tuning remains a limitation; the new test sample provides a separate assessment of the frozen recipes. Do not continue tuning against these test results.

## Inspect or reproduce

Import `counts-linear.json`, `counts-mlp.json`, `mean.json` or `transformer-two-head.json` from `output/imdb-final/models/`. All projects use seed 137 chosen in advance and include only training/validation reviews, saved optimizer settings and initial weights for Reset. The seed-137 checkpoint accuracy can differ from the table mean. Set batch 32 and up to 30 epochs manually. New runs reset optimizer moments. Generated datasets, models and reports remain git-ignored under output/imdb-final/.

After the preceding data/vocabulary and two-head studies:

```sh
node scripts/prepare-final-imdb-study.mjs
python3 scripts/train-data-vocab-study.py --root output/imdb-final/context256
node scripts/prepare-final-imdb-baselines.mjs
python3 scripts/train-final-imdb-baselines.py
node scripts/freeze-final-imdb-study.mjs
python3 scripts/evaluate-final-imdb-study.py
node scripts/report-final-imdb-study.mjs
```

Python requires PyTorch. Use a new output directory to reproduce; freezing refuses to overwrite an existing frozen.json, and test evaluation refuses to replace test-results.json. Preparation/reporting accept FINAL_IMDB_OUTPUT; Python scripts accept --root (use its context256 subdirectory for the context run). Raw IMDb defaults to /tmp/backprop-imdb/aclImdb; freezing accepts IMDB_ROOT. Prior-study directories are fixed as recorded in the preparation scripts.

The app now materializes word-count vectors on demand, removing the former dense-corpus allocation limit while retaining per-vector and text import limits. This allows the full 20,000-review / 4,000-word comparison to run in the app without caching 84 million dense count cells.

## Interpretation and verification

The context extension improved the transformer's mean validation accuracy from 81.5% to 85.7%. On the fresh test sample, linear word counts reached 86.5%, pooled embeddings 86.3%, the MLP on counts 85.5%, and the transformer 85.2%. These results do not support promising students that adding attention or a transformer will improve IMDb sentiment accuracy. Keep this comparison as an example of measuring whether added complexity helps; use the generative sequence to motivate contextual next-token prediction. Stop tuning this task against the now-observed test sample.

The working transformer recipe is 20,000 training reviews, training-only vocabulary 4,000, context 256, width 32, two 16-dimensional heads, AdamW learning rate 0.001, batch 32, weight decay 0.01, clip norm 1, no dropout, and validation-loss early stopping (patience 3, minimum improvement 0.001). Selected epochs were 2, 3, 3 across seeds, so retain early stopping instead of prescribing one fixed epoch count.

All four seed-137 checkpoints were restored, evaluated through the trace engine and round-tripped through the project's import parser. All 613 automated tests passed, including large-corpus lazy count vectors and native PyTorch parity. Lint and the production build passed. Browser WebGL checks used all 21,000 training/validation examples at vocabulary 4,000 and context 256 for the count MLP and transformer; both matched trace losses and completed finite forward/backward AdamW updates. This is a compatibility check, not a full browser training benchmark.
