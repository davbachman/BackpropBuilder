# Text curriculum pilot

This is an instructor feasibility run of the sequence after MLP classification. The models were assembled from ordinary editable blocks, trained on real data, and checked against PyTorch. They are instructor reference solutions, deliberately absent from the student model menu. Student instructions and Gradescope packages remain a separate design step.

The local run's datasets, trained project JSON, Python/notebook exports, measurements, and preview are in `output/text-curriculum/`. This directory is ignored by Git. Start with its `REPORT.md` for measured results; this guide explains the sequence and how to reproduce it.

## 1. IMDb: word-count sentiment classifier

Build a vocabulary from training reviews and convert each review into a `[1, V]` count vector. Connect counts → matrix product → bias → sigmoid → binary cross entropy. Then add a hidden layer and ReLU for the MLP comparison. The target is negative = 0, positive = 1.

Reference projects: `counts-linear.json`, `counts-mlp.json`. The pilot uses V = 1,000, including an unknown-token entry. It counts only the first 64 tokens of each review, using the same limit for subsequent sequence comparisons. This is a deliberately small browser experiment, not a full-length IMDb benchmark.

Inspect individual word weights using the Data tab. Permuting words does not change a count vector. More parameters or lower training loss need not improve held-out accuracy.

## 2. IMDb: token IDs, one-hot vectors, learned embeddings

Switch the same Dataset's representation to Token IDs. The vocabulary and train/test documents stay fixed. A review of length T now has IDs `[T]` and positions `[T]`.

For the explicit construction, connect IDs → **One-hot**, set its width to V, then multiply `[T,V]` by a trainable `[V,20]` parameter table. **Embedding lookup** computes the same result without allocating all the zeros. Its inputs are the embedding table first and token IDs second. Average along token axis 0 with keep-dimensions enabled, then use an MLP sentiment head.

Reference projects: `mean.json` and the equivalent `mean-onehot.json`. Repeated tokens contribute repeatedly to the same table row's gradient; this equivalence is checked numerically. Averaging embeddings still removes word order. Learned embeddings introduce a compact representation; they do not by themselves solve negation or syntax.

## 3. IMDb: order, positions, attention, transformer block

Use these controlled comparisons on the same vocabulary and documents:

| Reference project | Computation | What the comparison establishes |
| --- | --- | --- |
| `mean.json` | Embeddings → mean → classifier | Invariant to token permutation |
| `position-mean.json` | Token + position embeddings → mean → classifier | Still invariant: averaging separates into two sums |
| `attention.json` | Embeddings → self-attention → mean → classifier | Without positions or a mask, pooled attention is still invariant |
| `position-attention.json` | Token + position embeddings → self-attention → mean → classifier | Can use order, because attention combines token and position information |
| `transformer.json` | Positioned embeddings → pre-norm attention + residual → token-wise MLP + residual → final norm → pooling → classifier | A complete small, single-head bidirectional transformer |
| `imdb-causal-mask.json` | Same classifier weights with a causal mask inserted | A controlled masking experiment, not a separately optimized classifier |

Construct attention explicitly as Q = XWq, K = XWk, V = XWv; scores = QKᵀ/√d; row-wise softmax; attention output = probabilities × V. Each matrix and each operation remains inspectable. Position tables are learned parameters.

In Details, open **Word-order experiment**, reverse token order, and compare the prediction. The first three models must be invariant up to numerical roundoff. The others can change; that does not guarantee a meaningful sentiment improvement. Reversal changes natural language and is an architectural probe, not a new labeled evaluation set.

For masking, compare a prefix's contextual states before and after changing later tokens. Causal states cannot change when only the future changes. A sentiment classifier normally benefits from seeing the whole review, so use this as a bridge to prediction rather than claiming causal masking improves IMDb accuracy. Whole-review pooling can still use all tokens; causality is a property of each token's state.

## 4. Alice: next-token prediction

Use character tokenization to keep the output vocabulary small and make encoding/decoding transparent. Start with `alice-baseline.json`: each character embedding directly predicts the next character. Then build `alice-transformer.json`: learned token and position embeddings, one causal attention head, residual paths, normalization, token-wise MLP, and a vocabulary projection.

The pilot uses width 16, context 16, and 44 vocabulary entries including unknown. Each example's input is characters 0…15 and target is characters 1…16. Cross entropy averages over the 16 target positions. Pass **logits**, not softmax probabilities, to the loss.

The causal mask goes on the score matrix **before softmax**. A unit test changes future IDs and verifies earlier logits are unchanged. Corpus passages are split before window creation, and no window crosses a passage or train/test boundary.

The output projection is a trainable `[d,V]` matrix plus bias. It is not an inverse embedding. Optional weight tying connects the original `[V,d]` embedding parameter through **Tensor transform → Transpose (1,0)** and reuses it as the output matrix; the graph then accumulates gradients from both uses of that one parameter. The saved pilot uses an independent output matrix.

## 5. Alice: autoregressive generation

Open the trained Alice project. Details contains **Continue the text**. Enter a prompt, inspect next-token probabilities, or generate one or 40 tokens. Compare Most likely with Sample, changing temperature and top-k.

Every generated token comes from the student's current graph and weights. The app appends its decoded token and repeats, retaining the most recent context window and resetting position IDs within that window. Inference evaluates a copy without Loss and leaves the training graph/parameters unchanged. It rejects a prediction path that reads dataset targets. The unknown marker round-trips as one ID.

Greedy decoding tends to repeat. Sampling introduces variation. A short run on a small corpus learns spelling fragments and common short sequences; fluent prose is not the success criterion for this pilot.

## What the app now supports

- **Dataset → Text / reviews…** imports review CSV, plain text for next-token tasks, or prepared dataset JSON. CSV headers are `text,label,split` (also `review,sentiment`). Labels accept positive/negative, pos/neg, or 1/0. Explicit train/test splits are preserved; otherwise every fifth review within each class is held out before vocabulary fitting.
- Word-and-punctuation or character tokenization; optional lowercasing; training-only vocabulary; unknown IDs; count and token representations; position IDs; shifted next-token targets; text/token/vocabulary inspection.
- A differentiable **One-hot** path to a learned matrix, with no gradient into discrete IDs. Existing embedding, transpose, attention primitives, masking, and normalization are reused.
- Word/token labels in tensor inspection, including embedding rows and both axes of attention scores.
- Faster multi-epoch training that uses the same numerical operations and derivatives as step-by-step tracing. Fast and traced SGD match in regression tests.
- Save/import preserves the text preprocessing recipe and vocabulary. Python/notebook export includes a sibling dataset JSON containing prepared examples and text metadata. Exports train/evaluate those numerical examples; the generated Python does not include the browser's text-generation UI.

Text limits are 5 million characters, 10,000 documents/windows, vocabulary up to 4,096, and context up to 256. These are validation ceilings, not recommended transformer sizes. The existing parameter initializer supports at most 65,536 values per tensor; 1,000 × 20 fits. The pilot uses unpadded variable-length reviews and one review/window per SGD update. Numeric mini-batches do not apply to these sequence tensors. Attention costs grow quadratically with context length.

## Reproduce the run

Requires the app's usual Node dependencies. Obtain and extract the [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), and save [Project Gutenberg's Alice text](https://www.gutenberg.org/ebooks/11) as `output/text-curriculum/data/alice-original.txt`.

```sh
npm install
IMDB_ROOT=/absolute/path/to/aclImdb node scripts/pilot-text-curriculum.mjs
node scripts/report-text-curriculum.mjs
npm test
npm run lint
npm run build
```

`TEXT_PILOT_OUTPUT` changes the output directory. `TEXT_PILOT_MODELS` is a comma-separated subset for experiments; a subset run may load its precursor checkpoint from that output directory. The script samples balanced reviews deterministically from the official splits and saves the exact CSV, prepared text JSON, training reports, editable projects, and exports. It selects Alice training and held-out passages from disjoint portions of the book. Parameters and shuffle orders use deterministic seeds.

For independent PyTorch forward/backward checks, set `PYTORCH_TEST_PYTHON` to a Python executable with PyTorch installed before `npm test`. These checks otherwise skip cleanly. The pilot measurements use the held-out review sample to compare settings, so call it a validation set; do not present it as an untouched final benchmark.
