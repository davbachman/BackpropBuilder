# Remaining semester: 18 class days

Working plan revised around the instructor's first two days. Linear regression, gradient descent, and feature engineering have already been taught. Students build their own models from app operations; instructor models are demonstrations and feasibility references.

| Day | In class | Dataset and student work |
|---|---|---|
| 1 — Tuesday | Introduce the app; build linear regressors for its 1D and 2D toy datasets. | MPG: compare one input column, two input columns, then adding a squared feature. Keep the split fixed and compare training and validation error. |
| 2 | L2 for controlling overfitting; L1 for encouraging sparse coefficients. | MPG high-degree polynomial regression for L2; California housing: standardize all eight inputs, fit with L1, then refit the strongest three without a penalty. |
| 3 | Sigmoid, binary cross-entropy, and classification thresholds. | Breast-cancer diagnostic dataset: build a binary classifier. |
| 4 | Logits, softmax, and multiclass cross-entropy. | Iris: build a multiclass classifier. |
| 5 | Compose linear maps; demonstrate that the result is still linear. Introduce nonlinear activations and an MLP. | Concrete compressive strength: compare a linear model with a regression MLP. |
| 6 | Use an MLP for classification; distinguish hidden activations from output/loss choices. | 8×8 handwritten digits: compare linear and MLP classifiers. |
| 7 | Word-count vectors as numeric features; vocabulary fitting and unknown words. | IMDb: build a word-count sentiment classifier. |
| 8 | Introduce the last-location story bridge; show that identical counts can require different answers. Token IDs, one-hot interpretation, learned embeddings, and pooling. | Counts and pooled embeddings on the bridge. Embeddings alone are not expected to solve the order problem. |
| 9 | Preserve token order through fixed slots and an ordered embedding MLP. | Solve the last-location bridge; test matched reversed histories. Flattened slots preserve position without an added positional-embedding block. |
| 10 | Construction/debugging lab and controlled comparisons. | Complete the bridge comparison; check padding, embedding shapes, and reversal behavior. |
| 11 | Introduce bAbI Task 1: multiple people, distractors, and a question. Encode facts in word order, represent fact positions, and introduce question-directed Q/K/V attention. | Compare ordered/sentence MLPs with the attention model. The question attends to facts; this is cross-attention. |
| 12 | Attention construction and interpretation lab. | Train bAbI Task 1; inspect which facts receive weight for different questions. Compare against the same fact encoder with an MLP readout. |
| 13 | Switch to Alice: character tokenization, vocabulary logits, next-character cross-entropy, and autoregressive generation. | Build a bigram generator using the last character of each shared context. The vocabulary projection is learned, not an inverse embedding. |
| 14 | Use more context with an ordered embedding MLP. | Alice fixed-context MLP; compare with the bigram model on the same final-character targets. |
| 15 | Token-to-token self-attention, learned positional embeddings, and causal masking. | Alice causal attention model. Verify that earlier predictions cannot access future characters; no guaranteed gain over the MLP is claimed for attention alone. |
| 16 | Complete the transformer with residual connections, normalization, and a feed-forward block. | Alice transformer; compare with the MLP under the same final-character objective. |
| 17 | Train at every shifted position; distinguish extra supervision from architectural changes. Explore greedy decoding and sampling. | Alice all-position causal training and generation; compare models using the common final-position metric as well as their own objectives. |
| 18 | Integration lab, final evaluation, and model comparison. | Finish the generator and compare loss, generated text, runtime, and failure cases across the sequence. |

The rows describe a sequence of work, not 18 separate homework submissions. Adjacent days can contribute to one assignment. Lab days have a concrete construction milestone; homework can require a controlled comparison or transfer test beyond the in-class toy demonstration. Finishing construction quickly does not eliminate the comparison requirement.

## First two assignments: implementation notes

Day 1 revisits feature engineering only as app practice. Decide the required MPG columns and squared term so students' comparisons are interpretable. Adding a squared column should normally retain the original input columns and intercept. Scale features using training data only and preserve the same train/validation split across all variants.

Day 2's high-degree polynomial exercise is a distinct configuration from the completed app pilot. The pilot used a small MPG training subset and quadratic/interactions, and found an L2 benefit. A univariate high-degree example should still be piloted at the intended degree, scaling, sample size, and training budget before publishing numerical expectations. Scaling and adequate optimization matter: numerical instability should not be mistaken for statistical overfitting.

For L1 feature comparisons, put input features on comparable scales and exclude the intercept from penalties. The current app uses the L1 subgradient with SGD/AdamW; it does not guarantee exactly zero coefficients. A homework exercise can examine coefficient shrinkage and a declared near-zero threshold. If exact zero coefficients are the learning objective, implement and validate a proximal/soft-threshold update before assigning it. The [California housing pilot](STANDARDIZATION-HOUSING-PILOT.md) supports retaining median income, latitude, and longitude: the unregularized three-feature refit essentially matches the eight-feature baseline. Two features perform worse.

## Evidence and limits

See [Semester app pilot](SEMESTER-APP-PILOT.md) for exact settings, three-seed results, browser checks, and generated instructor files. The established progression includes concrete regression, digits classification, the last-location bridge, bAbI Task 1, and Alice. Attention alone did not reliably outperform the Alice MLP; the full transformer did. Character generation is operational but can repeat phrases and is not reliably fluent.

This revision changes the teaching plan only. It does not create student handouts, grading thresholds, or Gradescope packages. The new high-degree MPG exercise remains to be piloted; California housing is now piloted.
