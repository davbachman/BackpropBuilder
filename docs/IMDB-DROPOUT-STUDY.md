# Dropout and learning-rate experiment

Dropout is now an editable palette block. Its probability `p` can be changed in **Details**. During training it independently zeros each activation with probability `p`, and multiplies survivors by `1/(1−p)`. Backpropagation reuses that same mask. During evaluation it is the identity operation. The rate must satisfy `0 ≤ p < 1`; the default is 0.1 and zero disables dropout.

**Step** starts a training-mode forward pass on training examples; its local derivative shows the saved mask and scale. **Run one full training step**, trace-engine epoch training, and tensor-engine training also enable dropout. **Run forward**, loss reports, inference, and text generation disable it. If a forward pass has already been run, stepping backward uses that pass's cached mask. Project files preserve the rate and any current trace cache. PyTorch export uses `F.dropout(..., training=self.training)`; the existing tensor-optimizer export restriction still applies.

Tensor training draws a new mask at each dropout operation on each update. It uses a reproducible seed sequence starting fresh for each run. Trace training uses the browser's random generator. The native study uses PyTorch's seeded generator. These implement the same distribution and scaling, but do not produce identical masks or training trajectories across engines. Reported training loss is evaluated **without** dropout; the study's `onlineTrainingLossWithDropout` is a different quantity measured during updates.

## Experiment

The nine combinations used dropout probabilities **0, 0.1, 0.2** and learning rates **0.0001, 0.0003, 0.001**. Everything else stayed fixed: the original 5,000 training and 1,000 validation reviews; training-only vocabulary of 2,000; context 128; width 32; one attention head; batch 32; AdamW decay 0.01; global norm clipping 1. Runs allowed up to 30 epochs with validation-loss patience 3 and minimum improvement 0.001.

Dropout was inserted at three explicit locations: attention output before residual addition, feed-forward output before residual addition, and classifier ReLU before the final linear layer. Adding the nodes did not change any initial parameter values. The zero-dropout baseline at learning rate 0.0003 exactly reproduced the earlier validation losses for all three seeds.

The nine-setting screen used seed 137. Its two lowest-validation-loss settings were repeated with seeds 211 and 307, alongside the original baseline. A no-dropout control at the selected learning rate was then added to separate the learning-rate effect from dropout. This produced 17 total runs. All selection and comparison used validation metrics. The final-test split was not evaluated.

| Dropout | Learning rate | Mean validation loss | Mean validation accuracy | Accuracy range across three seeds |
| --- | --- | ---: | ---: | ---: |
| 0 | 0.0003 | 0.5249 | 75.7% | 74.2–76.9% |
| 0 | 0.001 | **0.5039** | **77.3%** | 76.8–77.8% |
| 0.1 | 0.001 | 0.5111 | 76.5% | 75.6–77.9% |
| 0.2 | 0.001 | 0.5069 | 76.6% | 75.5–78.4% |

The higher learning rate improved this model's mean validation performance. Dropout at these rates and locations did not provide a consistent additional benefit. The current working choice is learning rate **0.001 with dropout disabled**, retaining early stopping; all three seeds selected epoch 3. This is an exploratory validation result, not a new final-test accuracy, a guarantee for students, or evidence that dropout is generally ineffective. The previously reported transformer test accuracy of 76.5% used a different split and should not be compared directly to this table.

## Local outputs and reproduction

Generated files remain git-ignored in `output/imdb-dropout/`:

- `RESULTS.md` and `summary.json`: all nine screening results and the repeated comparisons.
- `protocol.json` and `selection.json`: settings and selection rules.
- `runs/`: per-epoch metrics and all 17 parameter checkpoints.
- `models/transformer-p0-lr0.001.json`: fixed-seed 137 checkpoint for the strongest mean setting.
- `models/transformer-p0.2-lr0.001.json`: corresponding dropout variant for inspection.

The four importable models include the training/validation data, initial parameters for reset, and optimizer configuration. Epoch count and batch-size form fields are not saved: set batch size 32 and up to 30 epochs explicitly. Each new run resets optimizer moments.

After preparing the [original controlled IMDb data](ACCELERATED-TEXT-TRAINING.md):

```sh
node scripts/prepare-dropout-study.mjs
python3 scripts/train-dropout-study.py
node scripts/report-dropout-study.mjs
```

The Python executable needs PyTorch. Preparation/report scripts accept `IMDB_STUDY_OUTPUT` and `DROPOUT_STUDY_OUTPUT`; training accepts `--source` and `--root`. Its `--confirmation-only` mode adds missing confirmation controls to an existing screening run; use it only with unchanged data, templates, and protocol. A normal invocation recomputes all runs. The study runs natively on CPU; browser training is independently checked on WebGL.
