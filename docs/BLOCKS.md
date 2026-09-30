# Block reference

[Guide home](../README.md) · [Get started](USAGE.md) · [Canvas](CANVAS.md) · [Datasets](DATASETS.md)

Choose any block from **Build** or the searchable menu on a blank canvas. Select it to edit its settings in **Details**. Connect outputs to inputs from left to right; a red block indicates a missing connection, invalid expression, or incompatible shape, with the exact error in **Details**.

| Block | Purpose and inputs |
| --- | --- |
| **Dataset** | A built-in or imported data source. Its labeled output ports provide feature columns and targets; see [Datasets](DATASETS.md). |
| **Input** | An editable constant or a named alias for an incoming dataset feature. A connected value overrides its stored value. |
| **Param** | A trainable scalar or tensor, used for weights, biases, filters, embeddings, or normalization parameters. Enter a shape such as `3, 1`, initialize, and edit individual values. |
| **Arithmetic** | An expression using `x1`, `x2`, etc. Supports `+`, `-`, `*`, `/`, numeric powers, parentheses, and elementwise broadcasting. |
| **Matrix product** | `[rows, inner] × [inner, columns] → [rows, columns]`. Feed a learned matrix from Param into the right port. |
| **Activation** | Applies the selected activation to its input. Use it after a weighted sum or matrix product. |
| **Target** | Passes the dataset's target to a loss and identifies that data as the target. You may also wire the dataset target directly to the loss target port. |
| **Loss** | First port: predictions; second port: targets. Choose squared error, mean squared error, mean absolute error, binary cross entropy, or cross entropy from logits. |
| **Embedding lookup** | Table `[vocabulary, width]` and integer IDs `[tokens]` produce `[tokens, width]`. |
| **Tensor transform** | Choose Reshape, Transpose, Slice, or Mean. Set shape, axis order, slice range, or averaging axes in Details. |
| **Concatenate** | Joins inputs along an axis. On axis 1, vectors `[n]` act as columns, so four `[112]` inputs produce `[112, 4]`. |
| **Softmax** | Converts scores into probabilities along the last axis. Useful for inspecting predictions; multiclass cross entropy takes raw logits directly. |
| **Causal mask** | Masks future positions in a square token-by-token score matrix before softmax. |
| **Layer norm** | Input `[tokens, width]`, learned scale `[width]`, learned bias `[width]`. Initialize scale to ones and bias to zeros. |
| **Convolution** | Image `[H, W, C]`, filters `[F, KH, KW, C]`, biases `[F]`. Valid convolution, stride 1; output `[H−KH+1, W−KW+1, F]`. |
| **Average pooling** | Averages 2×2 image patches with stride 2, separately for each channel. |

## Choosing a loss

For regression, use mean squared error or mean absolute error. For a binary classifier with probabilities, use binary cross entropy. For multiple classes, connect **raw logits** shaped `[examples, classes]` and integer class IDs shaped `[examples]` to **Cross entropy (logits)**. A sequence model uses token positions as examples. You can branch into Softmax if you want to display probabilities, but do not put Softmax between logits and this loss.

## Shape tips

- Reshape changes dimensions without changing the number of values. One `-1` can infer a dimension, for example `-1, 1` for a batch column.
- Transpose with empty axes reverses the order; `1, 0` swaps matrix rows and columns.
- Arithmetic broadcasts compatible dimensions. For example, adding a `[width]` bias to a `[batch, width]` matrix applies it to every row.
- A Param shape creates a tensor; changing only the shape field does not imply a matrix is filled with the displayed scalar. Use the initializer or edit its values in Details. The initializer accepts positive dimensions and up to 65,536 values.
- A convolutional filter's input channel count must match the image's channel count. Average pooling requires an image-like tensor.

## One-hot encoding

**One-hot** converts integer IDs `[T]` into rows `[T,V]`. Set Vocabulary size in Details. Multiplying by a trainable `[V,d]` matrix is equivalent to Embedding lookup; only the matrix learns, not the discrete IDs. Use lookup for normal training and one-hot to inspect this equivalence.

## Dropout

Connect one tensor input; output shape is unchanged. Set **Dropout probability** in Details (default 0.1, valid range 0 inclusive to 1 exclusive). During training, independently zero entries with probability p and scale retained entries by 1/(1−p). The backward pass uses the same mask. Run forward, reporting, inference, and generation use the unchanged input. Step on a training example and epoch training enable dropout. A probability of zero disables the block. See the [dropout experiment](IMDB-DROPOUT-STUDY.md) for placements and results.

## Standardize features

To name its output (for example, `u`), select the block and set **Output variable** in Details. Formulas and pseudocode use this name in both the assignment and downstream calculations. The block title stays separate. Clear the field to restore automatic naming; duplicate variable names receive a numeric suffix.

Fixed feature preprocessing: (x − training mean) / training standard deviation. Fit explicitly in the inspector after connecting inputs. Uses only training examples, treats constant columns with scale 1, and saves statistics for inference and export. Supports scalars or feature vectors on the last tensor axis. This is dataset feature scaling, distinct from Layer norm. See [housing exercise](STANDARDIZATION-HOUSING-PILOT.md).

## Finding blocks

The Build sidebar groups blocks into Core model, Features and tensors, Neural networks, Sequences and attention, and Images. Core model is expanded initially; category choices are remembered in this browser. Search blocks searches all categories, including collapsed ones. Clear the search to restore your expanded categories. Block operations and existing saved models are unchanged.
