# bAbI curriculum pilot

For the subsequent app integration, regularization, and browser validation, see [Semester app pilot](SEMESTER-APP-PILOT.md). The earlier study below retains its original scope and results.

104 native training runs; 60 validation-selected checkpoints frozen before held-out evaluation. Every confirmed comparison below contains seeds 137, 211 and 307. No pretrained models or library attention/transformer modules were used. These are native PyTorch prototypes, not yet browser-importable app models.

## Recommendation

Use this validated core: IMDb counts → the synthetic last-location bridge → ordered embedding MLP → official bAbI Task 1 → question-directed attention. The first bridge has 8,000 training, 2,000 validation and 2,000 test questions, with disjoint reversal-pair groups. Task 1 uses 8,000 training, 2,000 validation and 1,000 official test questions, split by complete story rather than individual question.

The bridge is intentionally simple: five moves by one person. Every history has a matched reversed history with identical word counts and a different answer. Thus a deterministic order-blind classifier cannot exceed 50% on these balanced pairs, irrespective of its optimizer or capacity. All full histories are unique across splits. Learned mean embeddings remain order-blind; an ordered MLP solves it.

Task 1 introduces multiple people, irrelevant moves, variable story lengths and a question naming the relevant person. A flat ordered MLP and an MLP using the same sentence encodings as the attention model both struggle; question-directed attention solves the official test set, including all 984 inputs not seen verbatim in training or validation. This is a measured advantage under the tested training budget, not a proof that an MLP cannot solve the task.

Optional extension: the controlled two-fact bridge has 9,000 training, 1,500 validation and 1,500 test questions. Three people carry distinct objects and each moves once; six facts are shuffled. Split by the complete underlying world assignment, keeping all three questions together. One attention pass and answer-only two-pass training are unreliable. Two passes with additional supporting-fact training labels reliably solve it. The labels teach the first pass to find the owner and the second to find the owner's location. At inference the model receives only facts and the question. This introduces an auxiliary training objective, not merely a larger model.

Do not assign full bAbI Task 2 as the next guaranteed success. It includes repeated moves, dropping and picking up objects; the tested models remained near 38% on its test questions. Longer answer-only training on the simpler two-fact bridge occasionally escaped a plateau, but only one of three three-pass runs solved it. That is not a dependable student recipe.

Full token-level transformer blocks also need caution: Task 4 looked perfect in the first seed but was less reliable across seeds. Its test set contains many exact repeats of training examples; novel-only results are shown separately. I would teach query/key/value attention using the verified question-directed model, then introduce full self-attention blocks and causal masking in the language-generation sequence. These experiments do not establish a reliable bAbI demonstration in which every added transformer component improves accuracy.

## Confirmed held-out results

| Task | Model | Training variant | Mean validation | Mean test | Test range | Novel-only test |
|---|---|---|---:|---:|---|---:|
| Last-location bridge (synthetic) | counts-mlp | standard | 35.6% | 35.4% | 35.2%–35.5% | 35.4% |
| Last-location bridge (synthetic) | mean | standard | 35.8% | 35.1% | 34.9%–35.2% | 35.1% |
| Last-location bridge (synthetic) | ordered-mlp | standard | 100.0% | 100.0% | 100.0%–100.0% | 100.0% |
| Official bAbI Task 1 | counts-mlp | standard | 39.7% | 41.1% | 39.6%–42.4% | 41.1% |
| Official bAbI Task 1 | mean | standard | 40.2% | 42.4% | 41.6%–43.4% | 42.3% |
| Official bAbI Task 1 | ordered-mlp | standard | 47.1% | 48.5% | 48.1%–49.2% | 48.3% |
| Official bAbI Task 1 | sentence-mlp | standard | 51.9% | 49.7% | 48.6%–50.8% | 49.4% |
| Official bAbI Task 1 | memory-1 | standard | 100.0% | 100.0% | 100.0%–100.0% | 100.0% |
| Official bAbI Task 4 | counts-mlp | standard | 49.5% | 50.3% | 48.7%–52.1% | 50.1% |
| Official bAbI Task 4 | ordered-mlp | standard | 85.4% | 84.5% | 76.6%–100.0% | 93.8% |
| Official bAbI Task 4 | transformer-1 | standard | 92.9% | 93.2% | 79.6%–100.0% | 96.9% |
| Official bAbI Task 4 | transformer-2 | standard | 92.6% | 92.9% | 78.8%–100.0% | 97.3% |
| Official bAbI Task 2 | memory-1 | standard | 40.5% | 37.6% | 35.9%–38.5% | 37.6% |
| Official bAbI Task 2 | memory-2 | standard | 39.7% | 38.1% | 36.4%–39.0% | 38.1% |
| Two-fact bridge (synthetic) | memory-1 | standard | 48.7% | 50.1% | 50.0%–50.3% | 50.1% |
| Two-fact bridge (synthetic) | memory-2 | standard | 48.8% | 49.0% | 48.7%–49.4% | 49.0% |
| Two-fact bridge (synthetic) | memory-2 | assisted | 100.0% | 100.0% | 100.0%–100.0% | 100.0% |
| Two-fact bridge (synthetic) | retrieval-1 | extended | 48.6% | 49.2% | 48.7%–49.8% | 49.2% |
| Two-fact bridge (synthetic) | retrieval-2 | extended | 47.9% | 48.8% | 47.6%–49.4% | 48.8% |
| Two-fact bridge (synthetic) | retrieval-3 | extended | 66.1% | 66.2% | 48.7%–100.0% | 66.2% |

Ranges describe initialization variation, not statistical confidence intervals. Accuracy is averaged across individual models, not an ensemble. All six answer classes are locations. Inspect full tables before comparing a 100-epoch run with a 30-epoch run or assisted training with answer-only training.

## What each model computes

- Counts: bag of word counts → linear classifier, or 64-unit ReLU MLP → six logits. The confirmed word-count baseline is the stronger MLP.
- Mean: learned 32-dimensional word embeddings → masked mean → 64-unit classifier.
- Ordered MLP: zero-padded embeddings in token order → flatten → 64-unit classifier.
- Token attention/transformers: two heads of width 16, learned absolute positions when enabled, explicit Q/K/V products and softmax. Transformers have pre-norm attention/residual and a 64-unit feed-forward/residual block; one or two blocks. Readout is mean pooling, with separate exploratory question-only pooling variants.
- Sentence MLP: each fact is encoded by a shared linear/ReLU map of its ordered word embeddings. Learned fact-position vectors are added. All fact vectors and a separately encoded question are flattened into an MLP.
- memory-1/2/3: the identical fact and question encoders, then query–key scores over fact vectors, softmax, and a weighted value sum. Each pass updates the question state with the retrieved vector. One pass is cross-attention from a question to facts, not a full self-attention transformer.
- retrieval-1/2/3: an additional memory-network prototype with adjacent key/value embedding sharing and output weights tied to answer-word embeddings. Its 100-epoch runs test whether short early stopping hid delayed learning.

All structured models use sentence boundaries and a separately marked question, available directly from the source data. That is an additional inductive bias compared with raw token models. The sentence-MLP control shares this preprocessing and encoders; differences are not solely due to receiving the question separately. Learned word/fact positions are used before attention; positions followed immediately by mean pooling are not presented as an order-aware solution.

## Training, splits and validation

Standard runs: AdamW 0.001, batch 64, decay 0.01, global norm clipping 1, no dropout, at most 30 epochs, validation-loss patience 5, minimum improvement 0.001. The best validation-loss checkpoint is restored. Extended runs use 100 epochs and patience 100, with the same optimizer. Assisted runs add 0.5 times the sum of the two fact-selection cross-entropies to answer cross-entropy; checkpoint selection uses answer loss only.

The study is exploratory: structured models, two synthetic bridges, question-only readout, longer training and auxiliary fact labels were added in response to validation failures. All exploratory outcomes are retained below. Final configurations and all seeds were recorded in frozen.json before test evaluation. No tuning followed test evaluation.

Vocabulary is fitted on training rows only; padding and unknown IDs are separate. Previous question answers, supporting-fact IDs, future story statements and test labels never enter model features. Split original training data by whole story. Exact duplicate input strings can still occur in the synthetic official corpus; report novel-only test results alongside standard test results.

| Task | Test questions | Novel questions | Exact training repeats | Excluded long questions |
|---|---:|---:|---:|---:|
| Last-location bridge (synthetic) | 2000 | 2000 | 0 | 0 |
| Official bAbI Task 1 | 1000 | 984 | 11 | 0 |
| Official bAbI Task 4 | 1000 | 259 | 682 | 0 |
| Official bAbI Task 2 | 994 | 994 | 0 | 6 |
| Two-fact bridge (synthetic) | 1500 | 1500 | 0 | 0 |

Task 2 excludes 75 of 10,000 train/validation questions and 6 of 1,000 test questions whose full inputs exceed 256 tokens. No story is truncated and no supporting fact is used to select which text to retain. Novel test inputs are those whose complete token sequences are absent from both train and validation.

## Files and reproduction

All generated data, native weights and results are git-ignored in output/babi-pilot/. Dataset CSVs are in data/; qa0 is the last-location bridge, qa1 the official single-fact task, and qa100 the controlled two-fact bridge. qa100-supported.csv also includes the two fact-selection labels, numbered from 1. The CSVs are pilot datasets, not yet compatible with the app's binary review importer. Native weights are in runs/*.pt. test-results.json contains frozen-checkpoint predictions, and error-examples.json contains actual failure cases.

Source: the bAbI en-10k archive mirrored at https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz; SHA-256 is recorded in protocol.json. Original task definitions: https://github.com/facebookarchive/bAbI-tasks. The two bridge datasets are generated here and are not official bAbI tasks.

Reproduction requires PyTorch. scripts/pilot-babi.py prepares the official training data and trains selected model families. scripts/prepare-babi-bridge.py and scripts/prepare-babi-twofact-bridge.py create the controlled datasets. Exact commands and stages are in reproduction.md. scripts/freeze-evaluate-babi.py refuses to overwrite a frozen study. Use BABI_PILOT_OUTPUT for a separate reproduction directory.

App integration still requires multiclass text answers, fixed padding/flattening for the ordered MLP, separate fact/question inputs and the optional supporting-fact objective. No browser execution, classroom runtime benchmark, student assignment or Gradescope package is claimed by this pilot. Six native behavioral tests cover parsing, leakage prevention, invariance, padding, gradients and matched readout initialization.

## All exploratory training runs

| Run | Best epoch | Epoch budget | Validation accuracy | Training accuracy |
|---|---:|---:|---:|---:|
| qa0-counts-mlp-seed137 | 7 | 30 | 35.9% | 35.4% |
| qa0-counts-mlp-seed211 | 30 | 30 | 35.8% | 35.4% |
| qa0-counts-mlp-seed307 | 7 | 30 | 35.4% | 35.5% |
| qa0-mean-seed137 | 29 | 30 | 35.6% | 35.5% |
| qa0-mean-seed211 | 29 | 30 | 35.9% | 35.5% |
| qa0-mean-seed307 | 29 | 30 | 35.9% | 35.4% |
| qa0-memory-1-seed137 | 4 | 30 | 100.0% | 100.0% |
| qa0-memory-2-seed137 | 7 | 30 | 100.0% | 100.0% |
| qa0-memory-3-seed137 | 2 | 30 | 100.0% | 100.0% |
| qa0-ordered-mlp-seed137 | 6 | 30 | 100.0% | 100.0% |
| qa0-ordered-mlp-seed211 | 6 | 30 | 100.0% | 100.0% |
| qa0-ordered-mlp-seed307 | 8 | 30 | 100.0% | 100.0% |
| qa0-sentence-mlp-seed137 | 4 | 30 | 100.0% | 100.0% |
| qa1-attention-no-position-seed137 | 30 | 30 | 40.0% | 43.8% |
| qa1-attention-position-seed137 | 19 | 30 | 52.4% | 54.7% |
| qa1-counts-linear-seed137 | 20 | 30 | 40.4% | 43.7% |
| qa1-counts-mlp-seed137 | 29 | 30 | 39.9% | 44.8% |
| qa1-counts-mlp-seed211 | 29 | 30 | 39.2% | 44.3% |
| qa1-counts-mlp-seed307 | 28 | 30 | 40.1% | 44.8% |
| qa1-mean-seed137 | 24 | 30 | 39.8% | 44.1% |
| qa1-mean-seed211 | 24 | 30 | 40.1% | 44.1% |
| qa1-mean-seed307 | 25 | 30 | 40.6% | 43.6% |
| qa1-memory-1-seed137 | 10 | 30 | 100.0% | 100.0% |
| qa1-memory-1-seed211 | 9 | 30 | 100.0% | 100.0% |
| qa1-memory-1-seed307 | 15 | 30 | 100.0% | 100.0% |
| qa1-memory-2-seed137 | 12 | 30 | 100.0% | 100.0% |
| qa1-memory-3-seed137 | 11 | 30 | 100.0% | 100.0% |
| qa1-ordered-mlp-seed137 | 5 | 30 | 47.1% | 63.7% |
| qa1-ordered-mlp-seed211 | 5 | 30 | 47.9% | 64.3% |
| qa1-ordered-mlp-seed307 | 5 | 30 | 46.2% | 63.4% |
| qa1-question-transformer-1-seed137 | 6 | 30 | 53.0% | 56.0% |
| qa1-question-transformer-2-seed137 | 6 | 30 | 51.9% | 56.0% |
| qa1-retrieval-1-seed137 | 13 | 30 | 100.0% | 100.0% |
| qa1-retrieval-2-seed137 | 8 | 30 | 100.0% | 100.0% |
| qa1-retrieval-3-seed137 | 7 | 30 | 100.0% | 100.0% |
| qa1-sentence-mlp-seed137 | 8 | 30 | 52.0% | 59.2% |
| qa1-sentence-mlp-seed211 | 10 | 30 | 52.0% | 60.7% |
| qa1-sentence-mlp-seed307 | 8 | 30 | 51.7% | 59.7% |
| qa1-transformer-1-seed137 | 12 | 30 | 51.0% | 57.6% |
| qa1-transformer-2-seed137 | 7 | 30 | 51.0% | 56.2% |
| qa100-memory-1-seed137 | 14 | 30 | 48.9% | 49.8% |
| qa100-memory-1-seed211 | 14 | 30 | 48.9% | 50.0% |
| qa100-memory-1-seed307 | 14 | 30 | 48.2% | 49.8% |
| qa100-memory-2-seed137-assisted | 14 | 30 | 100.0% | 100.0% |
| qa100-memory-2-seed137 | 19 | 30 | 48.1% | 50.0% |
| qa100-memory-2-seed211-assisted | 17 | 30 | 100.0% | 100.0% |
| qa100-memory-2-seed211 | 27 | 30 | 49.7% | 50.4% |
| qa100-memory-2-seed307-assisted | 10 | 30 | 100.0% | 100.0% |
| qa100-memory-2-seed307 | 14 | 30 | 48.7% | 50.0% |
| qa100-memory-3-seed137 | 14 | 30 | 48.7% | 49.8% |
| qa100-ordered-mlp-seed137 | 17 | 30 | 47.9% | 52.5% |
| qa100-question-transformer-1-seed137 | 18 | 30 | 48.3% | 50.5% |
| qa100-question-transformer-2-seed137 | 16 | 30 | 48.3% | 50.5% |
| qa100-retrieval-1-seed137-extended | 40 | 100 | 49.0% | 50.5% |
| qa100-retrieval-1-seed137 | 5 | 30 | 48.0% | 50.6% |
| qa100-retrieval-1-seed211-extended | 11 | 100 | 48.4% | 50.3% |
| qa100-retrieval-1-seed307-extended | 11 | 100 | 48.4% | 50.4% |
| qa100-retrieval-2-seed137-extended | 64 | 100 | 48.6% | 50.3% |
| qa100-retrieval-2-seed137 | 29 | 30 | 48.5% | 50.1% |
| qa100-retrieval-2-seed211-extended | 70 | 100 | 47.1% | 49.5% |
| qa100-retrieval-2-seed307-extended | 59 | 100 | 48.1% | 50.6% |
| qa100-retrieval-3-seed137-extended | 90 | 100 | 100.0% | 100.0% |
| qa100-retrieval-3-seed137 | 18 | 30 | 48.5% | 50.0% |
| qa100-retrieval-3-seed211-extended | 86 | 100 | 48.5% | 51.8% |
| qa100-retrieval-3-seed307-extended | 60 | 100 | 49.8% | 51.5% |
| qa100-sentence-mlp-seed137 | 15 | 30 | 48.6% | 50.4% |
| qa100-transformer-2-seed137 | 17 | 30 | 48.6% | 49.4% |
| qa2-attention-no-position-seed137 | 14 | 30 | 34.0% | 35.1% |
| qa2-attention-position-seed137 | 8 | 30 | 38.0% | 43.6% |
| qa2-counts-linear-seed137 | 19 | 30 | 35.6% | 35.1% |
| qa2-counts-mlp-seed137 | 17 | 30 | 34.3% | 38.1% |
| qa2-mean-seed137 | 17 | 30 | 35.1% | 35.7% |
| qa2-memory-1-seed137 | 13 | 30 | 40.7% | 41.6% |
| qa2-memory-1-seed211 | 13 | 30 | 41.0% | 42.3% |
| qa2-memory-1-seed307 | 16 | 30 | 39.9% | 42.4% |
| qa2-memory-2-seed137 | 13 | 30 | 39.8% | 42.8% |
| qa2-memory-2-seed211 | 10 | 30 | 39.7% | 42.4% |
| qa2-memory-2-seed307 | 10 | 30 | 39.7% | 42.2% |
| qa2-memory-3-seed137 | 12 | 30 | 39.8% | 43.3% |
| qa2-ordered-mlp-seed137 | 0 | 30 | 17.4% | 16.6% |
| qa2-question-transformer-1-seed137 | 3 | 30 | 36.7% | 45.0% |
| qa2-question-transformer-2-seed137 | 3 | 30 | 34.4% | 44.6% |
| qa2-retrieval-1-seed137 | 9 | 30 | 40.6% | 41.0% |
| qa2-retrieval-2-seed137 | 6 | 30 | 40.2% | 41.0% |
| qa2-retrieval-3-seed137 | 6 | 30 | 39.9% | 42.3% |
| qa2-sentence-mlp-seed137 | 2 | 30 | 20.5% | 35.4% |
| qa2-transformer-1-seed137 | 3 | 30 | 35.1% | 45.2% |
| qa2-transformer-2-seed137 | 2 | 30 | 34.8% | 41.9% |
| qa4-attention-no-position-seed137 | 26 | 30 | 51.0% | 52.3% |
| qa4-attention-position-seed137 | 11 | 30 | 56.6% | 58.1% |
| qa4-counts-linear-seed137 | 12 | 30 | 16.5% | 18.3% |
| qa4-counts-mlp-seed137 | 30 | 30 | 48.9% | 54.1% |
| qa4-counts-mlp-seed211 | 30 | 30 | 49.8% | 53.4% |
| qa4-counts-mlp-seed307 | 30 | 30 | 50.0% | 53.4% |
| qa4-mean-seed137 | 30 | 30 | 49.8% | 51.5% |
| qa4-ordered-mlp-seed137 | 10 | 30 | 78.1% | 80.6% |
| qa4-ordered-mlp-seed211 | 29 | 30 | 100.0% | 100.0% |
| qa4-ordered-mlp-seed307 | 10 | 30 | 77.9% | 80.6% |
| qa4-transformer-1-seed137 | 30 | 30 | 100.0% | 100.0% |
| qa4-transformer-1-seed211 | 23 | 30 | 100.0% | 100.0% |
| qa4-transformer-1-seed307 | 29 | 30 | 78.8% | 81.8% |
| qa4-transformer-2-seed137 | 14 | 30 | 100.0% | 100.0% |
| qa4-transformer-2-seed211 | 16 | 30 | 100.0% | 100.0% |
| qa4-transformer-2-seed307 | 13 | 30 | 77.7% | 80.3% |
