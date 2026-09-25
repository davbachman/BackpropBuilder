# Build models from scratch

[Guide home](../README.md) · [Canvas](CANVAS.md) · [Block reference](BLOCKS.md) · [Datasets](DATASETS.md)

These are connection recipes, not special app modes. Start with **File → New**, add each block from **Build** or the canvas menu, and wire the indicated tensors. Select blocks to set shapes, operations, and expressions in **Details**. Build and run one stage at a time so shape errors are easy to find. The [editable examples](EXAMPLES.md) show working versions of the same ideas.

## A numeric MLP

1. Choose a regression or classification Dataset and connect its feature columns. For multiple numeric columns, set **Concatenate** to axis 1: `k` columns of `[n]` become a feature matrix `[n, k]`. A single `[n]` feature can instead be reshaped to `[n, 1]`.
2. Add a **Param** for a weight matrix `[k, h]`. Connect the feature matrix and weight to **Matrix product**, producing `[n, h]`. Add a bias Param `[h]` through **Arithmetic** with `x1 + x2`; broadcasting applies it to every row. Follow with an **Activation** such as ReLU.
3. To make another layer, use a weight `[h, m]`, a bias `[m]`, another Matrix product and Arithmetic, and an Activation. Select the layer's calculations and group them if you want to copy or zoom into that unit.
4. End with a weight `[m, 1]` for a scalar regression output or `[m, classes]` for class logits. Connect the prediction and dataset target to **Loss**. Use mean squared error for regression or cross entropy from raw logits for multiclass classification.

For a tiny scalar neuron, replace the matrix products with arithmetic expressions such as `x1 * x2` and `x1 + x2`. **Step** in Train makes every multiplication, sum, loss, and gradient visible.

## One causal attention head

Let a sequence have `T` tokens and embedding width `d`. Connect token IDs to an **Embedding lookup** with a learned table `[vocabulary, d]`. You may also embed position IDs with a separate table and add token and position vectors.

1. Form **Q**, **K**, and **V** with separate Matrix product blocks and learned projection matrices. For one head their output width is `d_head`, so each is `[T, d_head]`.
2. **Tensor transform → Transpose** changes K to `[d_head, T]`. Matrix product `Q × Kᵀ` gives scores `[T, T]`. Multiply scores by a fixed Input of `1/√d_head` using Arithmetic.
3. Apply **Causal mask**, then **Softmax** to the scores. The mask prevents a token from attending to future positions. Matrix product the attention weights `[T, T]` with V `[T, d_head]` to get the head output `[T, d_head]`.
4. Select and group the head calculations. Click a wire or open a matrix in Details to inspect individual values.

For several heads, make a projection per head or slice wider Q, K, and V matrices along their width axis. **Concatenate** head outputs along axis 1 to return to width `d`, then apply an output projection `[d, d]`.

## A transformer block and decoder

Starting from a `[T, d]` token representation, a pre-normalization block follows this path:

```text
x → Layer norm → multi-head causal attention → output projection → Add(x) → r
r → Layer norm → Matrix product[d, m] → bias → Activation
  → Matrix product[m, d] → bias → Add(r) → next block
```

Each Layer norm takes the signal plus learned scale `[d]` (ones) and bias `[d]` (zeros). Use Arithmetic `x1 + x2` for both residual additions. Group the attention heads, MLP, and complete block; copy the block to stack more layers. After the last block, project `[T, d]` to raw vocabulary logits `[T, vocabulary]` and connect them with aligned next-token target IDs `[T]` to **Loss → Cross entropy (logits)**. Token and position IDs, and the next-token targets, come from a sequence Dataset.

The [miniature decoder example](../public/models/decoder.json) uses two such blocks. Its dimensions are intentionally small enough to inspect; you can expand its groups and compare the connections with your own graph.

## An 8×8 digit CNN

Choose **Handwritten digits · 8 × 8** on a Dataset block. Its image output is `[8, 8, 1]` and its target is a digit ID. A compact classifier is:

1. **Param** filters `[F, 3, 3, 1]` and bias `[F]` → **Convolution** with the image. The output is `[6, 6, F]`.
2. **Activation → ReLU**, then **Average pooling** → `[3, 3, F]`.
3. **Tensor transform → Reshape** to `[1, 9F]`. Matrix product with a Param `[9F, 10]` produces ten logits `[1, 10]`. Add a `[10]` bias.
4. Connect logits and the dataset digit target to **Loss → Cross entropy (logits)**. The target shape is `[1]`, containing an integer class ID.

Inspect the Convolution block in Details to see and edit filters and feature maps. Use **Test → Run inference** for held-out digit predictions and accuracy. The [CNN example](../public/models/cnn.json) contains a trained small classifier and a filter visualizer; the [dataset note](CNN-DATA.md) identifies its UCI source.
