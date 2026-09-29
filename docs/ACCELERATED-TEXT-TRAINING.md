# Batched text training and the controlled IMDb study

The Train sidebar offers **Tensor engine · batched text** for imported text datasets. Students still construct the same editable graph from individual operations. TensorFlow.js executes those operations and differentiates them in batches; it does not supply a pretrained model. The original trace engine still explains a single example's forward pass, derivatives, and SGD update.

## Using it

1. Open an imported-text graph, then select **Train → Training execution → Tensor engine**.
2. Choose **Auto** (WebGL, falling back to CPU), **AdamW**, and a batch size such as 32. Auto probes forward, backward, and optimizer kernels before training. WebGPU is available only by explicit selection and is experimental.
3. Enter an exact learning rate. The study below used 0.0003, weight decay 0.01, gradient norm limit 1, patience 3, minimum validation improvement 0.001, and at most 12 epochs.
4. Run epochs. Reports contain full training and validation loss each epoch, including epoch zero. With patience enabled, the run restores the checkpoint with the best qualifying validation loss. Stop also restores the best completed validation checkpoint. A nonfinite result fails the run without replacing the canvas parameters.

The imported split named `test` serves as **validation** for early stopping. Keep a separate final test outside this tuning process. Padding is added per batch; attention masks padded keys, pooling excludes padded positions, and language-model loss excludes padded targets. Padding never becomes a vocabulary token or a new trainable embedding.

Adam and AdamW retain their moments throughout one Run epochs operation. Each new run starts fresh moments; saving preserves model parameters and training settings, but not optimizer state, the epoch-run form fields, or an in-progress run. AdamW decays all parameters, including biases and normalization parameters. A gradient norm limit of zero disables clipping; patience zero disables automatic stopping and retains the final checkpoint of a completed run. Step and Run one full training step always demonstrate the original single-example SGD calculation.

## Scope and verification

This first tensor backend supports the curriculum's count classifiers, embedding/pooling classifiers, attention, transformer blocks, and causal next-token models. It supports matrix products, arithmetic, activations, embedding/one-hot, transpose, token-axis mean, feature concatenation, feature layer normalization, softmax, causal masking, BCE, and categorical cross entropy. Other graphs and transforms fail with an explanation; they can still use the trace engine. Python/notebook export explicitly rejects tensor training settings because its training loop currently implements SGD. Save a project JSON to preserve the tensor configuration, or explicitly switch to Trace with batch size 1 for an SGD export.

CPU batched losses and gradients are checked against the average of individual trace calculations on nine model variants, including unequal sequence lengths and causal attention. Independent native PyTorch tests also compare losses, gradients, and three clipped AdamW updates. Floating-point comparisons use tolerances because tensor arithmetic uses float32 while the trace engine uses JavaScript numbers.

A browser benchmark on the development machine measured 20 transformer updates after warmup, batch 16, context up to 64, width 16. Tensor CPU took 19.58 seconds and WebGL took 1.07 seconds (about 18×). Maximum initial prediction and gradient differences were about 6×10⁻⁸; both ended at loss 0.159615. This measures the update loop, not imports, compilation, epoch reporting, or a complete classroom training run. Production also checks a scalar loss each batch. Speed varies with the graph, browser, and hardware. WebGPU passed initial checks but produced nonfinite values during extended training here, so Auto excludes it.

Import ceilings are 80 million text characters, 60,000 documents/windows, 8,192 vocabulary entries, and 256 context tokens. Word-count datasets retain token IDs and materialize dense count vectors only when accessed, avoiding a corpus-sized dense count matrix. Parameter initialization permits up to 1,048,576 values per tensor. These are validation limits, not recommended classroom settings; attention memory grows quadratically with context.

## Controlled IMDb results

The study fixes 5,000 balanced training reviews, 1,000 validation reviews from the official training split, and 1,000 fresh final-test reviews from the official test split, excluding the previous pilot's test reviews. The vocabulary uses training text only: 2,000 entries, first 128 tokens per review, embedding width 32. All models share the data and training order; seeds 137, 211, and 307 change initialization. Checkpoints use validation loss only. The final test was evaluated after checkpoint selection.

| Model | Mean final-test accuracy | Range across three seeds |
| --- | ---: | ---: |
| Word-count linear classifier | 79.7% | 79.5–80.2% |
| Mean embeddings + classifier | 78.0% | 76.7–78.9% |
| Attention without positions | 80.1% | 79.7–80.6% |
| Positional attention | 79.1% | 78.0–79.6% |
| Transformer | 76.5% | 76.1–76.7% |

These runs establish stronger baselines; they do **not** demonstrate a reliable accuracy benefit from positional embeddings or a transformer. Nor are these optimized settings: counts and mean-embedding models reached the 12-epoch cap. Several changes differ from the original pilot, so the absolute improvement cannot be attributed to AdamW alone. The final test is now observed and must not become a tuning set.

The 15 runs used a native PyTorch CPU interpreter of the same primitive graphs, verified against the browser implementation. Their timings are not browser timings. Locally generated data, manifests, metrics, and instructor projects remain git-ignored under `output/accelerated-imdb/`. `models/*.json` contains five importable checkpoints at the fixed seed 137, with the training/validation data included. All seeds' weights and reports are retained under `runs/`; final-test rows remain outside importable projects.

## Reproduction

Download and extract the official [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). With Node dependencies installed and Python with PyTorch available:

```sh
IMDB_ROOT=/path/to/aclImdb node scripts/prepare-imdb-study.mjs
python3 scripts/train-imdb-study.py
node scripts/report-imdb-study.mjs
PYTORCH_TEST_PYTHON=/path/to/python3 npm test
```

`IMDB_STUDY_OUTPUT` changes the output directory for the JavaScript preparation/report scripts; the Python runner accepts `--root`. The preparation script writes exact source filenames and split assignments to `manifest.json`. It also records the initial trace-engine loss so native runs verify graph agreement before training. `RESULTS.md`, `summary.json`, and per-run JSON reports provide the full local results. Generated data and models must stay out of Git.

To repeat the original browser benchmark, generate the earlier text-curriculum pilot data as described in its guide, start `npm run dev`, and open `/NeuralCanvas/scripts/browser/tensor-benchmark.html` on that server. Click **Run GPU training checks**. It uses 32 pilot reviews, deliberately shortens some to exercise padding, and compares CPU/WebGL/WebGPU with the same initial parameters. GPU checks may take tens of seconds; a backend error is a failed result, not a valid timing.

The follow-up [dropout and learning-rate experiment](IMDB-DROPOUT-STUDY.md) adds a Dropout block and tests nine settings. Its three-seed validation comparison favors learning rate 0.001 without dropout; these validation figures are separate from the original final-test table above.

The subsequent [training-size and vocabulary experiment](IMDB-DATA-VOCAB-STUDY.md) compares nine combinations over three seeds, with fixed validation reviews and training-only vocabularies.
