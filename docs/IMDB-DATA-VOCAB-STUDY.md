# IMDb training-data and vocabulary study

This follows the [dropout study](IMDB-DROPOUT-STUDY.md), retaining learning rate 0.001 and disabling dropout. The grid crosses 5,000 / 10,000 / 20,000 balanced training reviews with 2,000 / 4,000 / 8,000 vocabulary entries, including the unknown token. Every cell uses seeds 137, 211 and 307.

Training subsets are nested and exclude the same 1,000 validation reviews used previously. Vocabularies are fitted only on each training subset. Preparation checks filenames, split sizes, nesting, and exact reproduction of the original 5,000-review / 2,000-word tokenized data. The final-test split is not evaluated. All nine combinations are compared by mean validation loss; accuracy is reported separately.

The transformer remains width 32, one attention head and context length 128. Training uses AdamW, batch 32, learning rate 0.001, decay 0.01, global gradient norm clipping 1, up to 30 epochs, validation-loss patience 3 and minimum improvement 0.001. The native PyTorch runner interprets the editable graph using primitive operations. It checks initial loss against the app's trace engine. Browser and native numeric trajectories can differ.

This is a practical recipe comparison rather than an equal-compute experiment. Larger datasets get more optimizer updates per epoch. Larger vocabularies change token identities and initial weights and increase the embedding parameter count. Three initializations help expose run-to-run variation; their ranges are not confidence intervals. Continued tuning on the same validation reviews means these results are exploratory, not new final-test accuracy estimates.

## Results

Mean validation accuracy across three initializations:

| Training reviews | 2,000 words | 4,000 words | 8,000 words |
| ---: | ---: | ---: | ---: |
| 5,000 | 77.3% | 75.8% | 75.6% |
| 10,000 | 78.5% | 78.6% | 79.0% |
| 20,000 | 80.5% | 81.2% | 81.1% |

The selected setting is **20,000 training reviews / 4,000 vocabulary entries**: mean validation loss 0.4052, accuracy 81.2%, with accuracy 80.4–81.7% across seeds. The original control is 77.3% (loss 0.5039), reproduced exactly. This is a gain of 3.9 percentage points on the same validation reviews.

More data improved mean accuracy at every vocabulary size. Larger vocabularies did not help with only 5,000 training reviews. At 20,000 reviews, 8,000 entries performed almost identically to 4,000 (81.1% vs 81.2%; loss 0.4080 vs 0.4052), while using 268,985 parameters instead of 140,985. The small difference between those two settings is not evidence of a reliable accuracy advantage; 4,000 is the more economical working choice.

All three 20,000-review runs selected epoch 2 at every vocabulary size. Later epochs generally worsened validation loss despite improving training fit, so simply adding epochs is not supported by this experiment. The imported selected checkpoint uses seed 137 and has 80.4% validation accuracy; 81.2% is the mean across three runs.

Selected project: `output/imdb-data-vocab/models/transformer-n20000-v4000.json`. Original control: `output/imdb-data-vocab/models/transformer-n5000-v2000.json`.

## Reproduce

After preparing the [original IMDb study](ACCELERATED-TEXT-TRAINING.md) and dropout comparison:

```sh
node scripts/prepare-data-vocab-study.mjs
python3 scripts/train-data-vocab-study.py
node scripts/report-data-vocab-study.mjs
```

Python requires PyTorch. The extracted official IMDb dataset defaults to `/tmp/backprop-imdb/aclImdb`, overridden by `IMDB_ROOT`. Preparation accepts `IMDB_STUDY_OUTPUT` for the preceding split artifacts. Preparation/reporting accept `DATA_VOCAB_OUTPUT`, while training uses `--root`. Reporting checks baseline reproduction against `output/imdb-dropout/`.

Generated artifacts remain git-ignored in `output/imdb-data-vocab/`: split manifests and prepared datasets in `data/`, initial graphs in `templates/`, 27 checkpoints and epoch histories in `runs/`, and importable baseline/selected projects in `models/`. `protocol.json`, `summary.json`, and `RESULTS.md` preserve the settings and full comparison. Importable projects use fixed seed 137 and include training/validation reviews plus initial parameters for Reset. Set batch 32 and up to 30 epochs explicitly; those form fields are not saved. New training runs reset optimizer moments.

## App compatibility

The trace engine's tensor limit now matches the existing bounded initializer and one-hot limit: 1,048,576 values. Previously, the initializer could create an 8,000 × 32 embedding table but trace evaluation rejected it at 200,000 values. A regression test checks large-table gathering and accumulation for repeated tokens, and oversized-operation checks still enforce the bound. The 8,000-word transformer was also checked on WebGL: its initial loss matched trace within 2e-9 and an AdamW update produced finite parameters and loss.

The follow-up [two-head attention study](IMDB-TWO-HEAD-STUDY.md) keeps the selected data and vocabulary sizes and total model width fixed.
