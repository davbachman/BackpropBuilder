# Semester app pilot

The core curriculum now runs as editable app graphs: numeric regression/classification → MLPs → IMDb counts (previous study) → ordered story embeddings → question-directed attention → Alice character generation. L1/L2 are implemented in the Loss inspector and both training engines. No pretrained networks or library transformer modules are used.

This report distinguishes full native training, browser checkpoint/gradient checks, and full browser training. The three-seed tables below are native runs of the exact exported app graphs. They are not 66 independent full browser retrainings.

## Held-out results

All rows average initialization seeds 137, 211 and 307; models are not ensembled. The 66 checkpoint configurations were frozen before this final evaluation. Early tabular and Alice final splits were withheld during tuning. The bAbI/bridge test sets had already been used in the earlier native study, so their reuse verifies app compatibility rather than providing a new untouched benchmark.

| Model / dataset | Test examples | Mean test result | Range across seeds |
|---|---:|---:|---:|
| mpg-linear | 59 | 3.802 RMSE | 3.498–4.259 |
| mpg-quadratic | 59 | 2.739 RMSE | 2.659–2.835 |
| cancer-linear | 86 | 95.0% accuracy | 93.0%–96.5% |
| concrete-linear | 155 | 11.358 RMSE | 11.284–11.428 |
| concrete-mlp | 155 | 5.662 RMSE | 5.491–5.836 |
| digits-linear | 270 | 95.8% accuracy | 95.6%–96.3% |
| digits-mlp | 270 | 97.9% accuracy | 97.4%–98.1% |
| qa0-counts-mlp | 2000 | 35.4% accuracy | 35.2%–35.6% |
| qa0-mean | 2000 | 35.1% accuracy | 34.9%–35.2% |
| qa0-ordered-mlp | 2000 | 100.0% accuracy | 100.0%–100.0% |
| qa1-ordered-mlp | 1000 | 47.5% accuracy | 46.9%–48.2% |
| qa1-sentence-mlp | 1000 | 51.0% accuracy | 50.4%–51.7% |
| qa1-memory-1 | 1000 | 100.0% accuracy | 100.0%–100.0% |
| alice-bigram | 1359 | 2.343 next-character CE | 2.343–2.344 |
| alice-mlp | 1359 | 2.180 next-character CE | 2.175–2.183 |
| alice-attention | 1359 | 2.199 next-character CE | 2.174–2.224 |
| alice-transformer | 1359 | 1.987 next-character CE | 1.967–2.004 |
| iris-faster | 23 | 100.0% accuracy | 100.0%–100.0% |
| alice-alltokens | 1359 | 1.732 next-character CE | 1.714–1.746 |
| mpg-small-l2-0 | 59 | 3.760 RMSE | 3.124–4.146 |
| mpg-small-l2-0.1 | 59 | 3.317 RMSE | 3.083–3.638 |
| mpg-small-l2-1 | 59 | 3.195 RMSE | 3.092–3.276 |

RMSE is in original units: miles per gallon or concrete compressive strength (MPa); lower is better. Alice cross-entropy is measured at the final context position for every model; lower is better. The all-token transformer also has all-position test loss 1.804 and accuracy 46.8%; its final-position accuracy is 48.9%. Those two evaluations have different denominators. Small test sets, especially Iris, do not justify claims of universal accuracy. Seed ranges describe initialization variation, not statistical confidence intervals.

## What to teach with these results

1. Introduce the app using the existing one-dimensional regression example, then use MPG with six numeric features. Add feature squares: mean test RMSE falls from 3.80 to 2.74 mpg.
2. Keep a short regularization lesson. The full MPG dataset shows little benefit from penalties; use the supplied 40-row training subset and all quadratic/interactions (27 features) to expose overfitting. L2 λ=1 reduces mean test RMSE from 3.76 to 3.19 mpg and makes validation performance less sensitive to initialization. λ=0.1 also helps on average. This is a deliberately data-limited exercise, not a claim that λ=1 is generally optimal. L1 implementation is verified, but no dependable L1 advantage is claimed.
3. Binary logistic classification: breast-cancer diagnostic features; multiclass softmax: Iris. Use Iris learning rate 0.01, not the exploratory 0.001 run. Numeric datasets are educational prediction examples.
4. Explain why stacked linear maps remain linear, then introduce ReLU and a 64-unit MLP. Concrete strength gives a clear regression improvement (11.36 → 5.66 MPa RMSE); 8×8 digits gives a classification improvement (95.8% → 97.9%).
5. Use the previously piloted IMDb word-count classifier as a strong order-blind baseline. Learned embeddings do not have to improve accuracy to be worth introducing.
6. Give the last-location bridge: histories with the same words but different order require different answers. Counts and pooled embeddings remain near 35%; the ordered embedding MLP reaches 100%. Flattening ordered slots preserves position; merely adding position vectors and immediately averaging does not demonstrate useful order sensitivity.
7. Move to official bAbI Task 1: multiple people, distractors and a question. The ordered MLP drops to 47.5%, and a sentence-encoder MLP reaches 51.0%. Query/key/value attention over the same ordered fact encodings reaches 100%. This is question-to-fact cross-attention, not yet a full self-attention transformer. Sentence boundaries and the separately marked question are explicit inputs; both structured comparison models receive them.
8. Move to Alice next-character prediction: bigram → fixed-context MLP → causal self-attention → full transformer. Attention alone is not a success step here (2.199 vs MLP 2.180 CE). The full transformer improves to 1.987 with the same final-token objective. Training at every permitted shifted position gives 1.732 final-position CE. Explain that the last change supplies more supervised targets per context as well as efficient training; do not attribute that gain solely to architecture.

The evidence supports this sequence, but not a promise that every added component independently increases accuracy. Use bAbI to motivate attention, and use Alice to assemble positions, causal masking, residual paths and the feed-forward block into a working generator. At context length 16 and width 32, generated text is an educational character model, not reliably coherent prose.

## App changes and use

- Select a Loss block → Details → Parameter regularization. Choose None/L1/L2, strength λ, and the parameter checkboxes. L1 adds λ Σ|w|; L2 adds λ/2 Σw². Shared parameters are counted once. Bias blocks are excluded by default; the supplied recipes explicitly select weight matrices. L1 uses subgradient zero at zero. The trace shows the additional parameter-gradient contribution.
- Dataset evaluation reports prediction/data loss; the training objective includes the penalty. AdamW weight decay is separate. The regularization exercises set decay to zero so students can isolate L1/L2.
- Numeric CSVs support up to 128 feature columns plus a target. An optional split column containing train/test preserves exact row assignments and is removed from model inputs. In these pilot files, app “test” rows are validation rows used for early stopping; the final test examples are stored separately.
- Multiclass text CSVs use text,label,split; structured stories additionally use question, with newline-separated facts inside the quoted text cell. The importer fits vocabulary and answer labels from training rows, distinguishes unknown and padding tokens, and rejects examples exceeding configured lengths instead of silently removing facts.
- Fixed token padding enables ordered flattening. Structured story inputs expose fact IDs, question IDs, positions, and padding masks. Embedding lookup supports these higher-rank inputs. Accelerated training now handles numeric regression/classification and structured stories as well as review text.
- Alice fixed-context models can predict just the next final character or train on every shifted target. Generation decodes IDs and pads short prompts. Everything is composed of editable palette operations.

## Training settings

| Recipes | Learning rate | Batch | Maximum epochs | Early stopping | AdamW decay |
|---|---:|---:|---:|---|---:|
| MPG / cancer / concrete / digits | 0.001 | 32 | 120 | patience 15, min Δ 0.0001 | 0 |
| Iris (faster) | 0.01 | 32 | 200 | patience 15, min Δ 0.0001 | 0 |
| Small MPG, λ=0 / 0.1 / 1 | 0.003 | 32 | 400 | patience 30, min Δ 0.0001 | 0 |
| Bridge / bAbI Task 1 | 0.001 | 64 | 30 | patience 5, min Δ 0.001 | 0.01 |
| Alice | 0.001 | 64 | 35 | patience 5, min Δ 0.001 | 0.01 |

All use global gradient clipping at norm 1, restore the best validation-loss checkpoint, and use no dropout. Text embedding width is 32; MLP hidden width is 64. Alice context is 16 characters with stride 8. Epoch and batch controls should be set from this table when retraining; they are not both stored in the project JSON. These are tested starting settings, not a search proving globally optimal hyperparameters.

## Data and files

Generated datasets, trained instructor models, and results are git-ignored under output/semester-pilot/. The tracked scripts and this report describe reproduction.

- models/*.json: import through File → Import. Models contain trained parameters plus initial values for Reset; students should receive the CSV and build their own graph, not receive these instructor checkpoints.
- data/mpg.csv, cancer.csv, iris.csv, concrete.csv, digits.csv: train/validation numeric CSVs. Scaling is fitted on training rows only. data/*.json retains source indices, scalings and the separate final-test partition. MPG excludes missing horsepower rows and uses cylinders, displacement, horsepower, weight, acceleration and model year; origin/name are omitted. Other feature order follows the source dataset. Class labels class0/class1 mean malignant/benign for cancer; Iris class0/1/2 means setosa/versicolor/virginica; digits classN is digit N.
- data/mpg-small.csv: 40 training rows and the same 59 validation rows; scaling is refitted on the 40 rows. Its graph creates all quadratic/interactions. data/mpg-small-scaling.json records that transformation.
- data/qa0-*.csv and qa1-*.csv: bridge and official single-fact question tasks, 8,000 training plus 2,000 validation examples each. Final tests contain 2,000 and 1,000 respectively. Original bAbI splits group whole stories; bridge reversal pairs stay in the same split.
- data/alice.json: prepared train/validation character corpus. Paragraphs of at least 80 characters are normalized and partitioned in book order: first 80% training, next 10% validation, last 10% final test. No context crosses a paragraph boundary. Vocabulary is fitted on training paragraphs. There are 12,719 training and 1,186 validation windows.
- templates/*.json: initial graph and exact training budget; numeric/*.json: app-preprocessed rows; runs/*.json: native validation history, weights and parity references; browser/*.json: actual WebGL checks/training; frozen.json and final-results.json: frozen configuration hashes and final metrics.

Sources: UCI Auto MPG and Concrete Compressive Strength; scikit-learn bundled breast cancer, Iris and 8×8 handwritten digits; the existing bAbI pilot archive and controlled last-location generator; the existing Project Gutenberg Alice text. This pilot does not silently replace digits with MNIST.

## Reproduction and safeguards

Use a fresh checkout/output directory for a new study. Preparation and training scripts refuse to modify an already frozen study. They depend on the prior bAbI pilot data in output/babi-pilot/ and Alice source in output/text-curriculum/data/alice-original.txt; see BABI-PILOT-REPRODUCTION.md and TEXT-CURRICULUM.md for those inputs. Install Python dependencies numpy, scipy, scikit-learn, xlrd, openpyxl and torch in a virtual environment; run npm install for the app.

Run these stages in order (python denotes that virtual environment):

```sh
python scripts/prepare-early-pilot.py
node scripts/prepare-semester-pilot.mjs
node scripts/refine-semester-pilot.mjs
node scripts/prepare-regularization-pilot.mjs
node scripts/replicate-qa-app-pilot.mjs
python scripts/train-semester-pilot.py
python scripts/check-semester-updates.py --all
node scripts/export-semester-models.mjs
python scripts/freeze-semester-pilot.py
node scripts/prepare-semester-final.mjs
python scripts/evaluate-semester-final.py
node scripts/serve-semester-pilot.mjs
```

Open the local pilot page printed/served on port 5174 at /NeuralCanvas/scripts/browser/semester-pilot.html. Check all recipes verifies representative seeds and all full-MPG penalty settings. Train from initialization executes the real app engine. The native interpreter is in scripts/lib/curriculum_tensor_model.py; instructor graphs are in src/test/curriculumModels.ts. Do not rerun preparation stages over an existing exploratory manifest, which appends refinement runs. Final evaluation refuses to overwrite its result.

## Verification and limits

Browser import succeeded after enabling the Chrome extension’s file access. In the main app, selecting L1 and changing λ updated the displayed Loss formula; the L2 parameter selection and fixed CSV split controls were also inspected. Every exported project is parsed back through the app importer.

35 WebGL checkpoint checks cover every core architecture, refined Iris, Alice all-token training, and the regularization strengths. Checks compare initial loss and gradients, predictions after three AdamW updates, full validation loss, JSON round trips, and Alice decoding. Initial loss tolerance is 1e-4; gradient/prediction/validation tolerances are 2e-4.

A bAbI memory-model weight coordinate differed by 0.00159 after three updates despite initial gradients agreeing to 6e-8. A ReLU boundary produced a near-zero-gradient sign difference that AdamW amplified. Predictions still agreed to about 4e-6, so the update gate checks predictions rather than requiring identical coordinates. The diagnostic remains in browser/qa1-memory-1-137-update-diagnostic.json. This is numerical agreement, not bit-identical training.

Full WebGL retraining of qa1-memory-1-137 completed 14 epochs, restored epoch 9, and reached 100% validation accuracy in 141.97 seconds on this computer. Other browser timings and final checks are recorded below. Runtime is hardware/browser specific.

These are instructor feasibility models, not finished student handouts or Gradescope packages. Manual assembly of many scalar numeric CSV columns is still verbose, especially for digits. No new multi-hop bAbI exercise is claimed. The full earlier IMDb study remains the evidence for IMDb; it was not repeated here.

Final software verification: **631 tests in 56 files passed**, including native PyTorch export/parity tests. TypeScript/Vite production build and ESLint passed. Vite reports its existing large-chunk advisory. Generated datasets and instructor checkpoints remain ignored by Git.

Full WebGL Alice training (`alice-alltokens-137`) completed 35 epochs in **548.24 seconds**, restored epoch 33, and produced validation cross-entropy 1.8262 with approximately 46.7% token accuracy. The native and browser runs use different seeded minibatch shuffles, so their full trajectories are not expected to coincide. Full WebGL small-MPG L2 training (`mpg-small-l2-1-137`) completed 168 epochs in **24.29 seconds**, restored epoch 138, and achieved validation data MSE 0.2533 (the objective additionally includes the penalty).

Main-app testing exposed a synchronous inference freeze on the large story dataset. Tensor-engine inference now evaluates in minibatches, yields between batches, and decodes predictions into the existing report table without changing weights. Regression loss excludes penalties and does not report a spurious classification accuracy. Tests compare the decoded rows, losses and accuracies against trace evaluation for QA, numeric regression/binary classification and all-position language prediction. The main app then evaluated all **2,000 bAbI validation questions at 100% accuracy** without freezing. The data inspector now shows each padded fact and question separately.

The main-app Alice project also imported successfully and generated 40 additional characters through its generation controls. Greedy decoding repeated a short phrase; the model is operational, but fluent prose is not an outcome established by this pilot. The final local app was left open on the small-MPG example with L2 λ=1 and only the weight matrix selected. Screenshots are saved in output/semester-pilot/browser/regularization.png and babi-inference.png. An instructor file index is in output/semester-pilot/START-HERE.md.
