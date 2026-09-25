# Datasets

[Guide home](../README.md) · [Get started](USAGE.md) · [Blocks](BLOCKS.md) · [Training](TRAINING.md)

Add a **Dataset** block and choose its **Source** on the block or in **Details**. The block's output handles are labeled with feature and target names. Connect features into the model through **Input** blocks or directly to operations. Connect the desired target to a **Target** block, then to the second input of **Loss**; a direct dataset-to-loss target connection also works.

Selecting a dataset shows its **Train / test split**, **Examples**, and, where applicable, **Output mode** in Details. **One example** is useful for tracing a calculation. **Numeric batch** sends an entire numeric split through the graph. Image and token-sequence sources provide complete tensor examples one at a time. Changing the selected example changes what you inspect on the canvas; multi-epoch training still uses the training split, and **Test → Run inference** evaluates the selected test or training split without updating weights.

## Included sources

| Family | Sources and use |
| --- | --- |
| Regression | Line and cubic functions in one dimension; a plane in two dimensions. Good for an initial neuron or MLP. |
| Classification | Threshold, circle versus center, parabola boundary, XOR, and neuron basics. Build a decision boundary and inspect errors. |
| Image | 500 normalized UCI 8×8 handwritten digits: 400 training and 100 held-out images, with digit IDs 0–9. This is not MNIST. |
| Token sequences | Color cycle and counting, with token IDs, position IDs, and aligned next-token targets. Used for the small transformer. |
| Attention and probabilities | Query/key/value message lookups and class scores for inspecting attention and softmax. |

The **Train / test split** selector offers the source's default split or 60%, 70%, 75%, 80%, and 90% training splits. The Details panel shows how many examples each side contains.

## Import your own CSV

Choose **Custom CSV…** as a Dataset source, then select a local `.csv` file. The file remains in the browser and is included in a saved project JSON; importing the saved project does not require the original CSV. The parser supports a header row, quoted fields, at least two data rows, 2–32 columns, up to 10,000 rows, and a 10 MB file. Feature columns must be numeric. A classification target may contain text labels.

Each output handle uses its CSV column name when headers are present. Wire the target column to **Target** or directly to the target input of **Loss**. That wiring tells the app which column is the target; until then the preview uses the last column or the sole text-label column. In Details you can change **Task** between regression, binary classification, and multiclass classification, toggle **First row contains headers**, and **Replace CSV file**. Invalid feature values or an incompatible task show an error there.

For several numeric columns, connect each column to **Concatenate** with axis 1. Vectors of shape `[n]` are treated as `[n, 1]` for this operation, so four columns become `[n, 4]` without four separate reshape blocks. Other operations keep their own shape rules; see [Block reference](BLOCKS.md).

## What gets reported

The prediction input to **Loss** and its target input define the model's supervised output. **Reporting** can plot a one- or two-input prediction graph when that path is connected and valid. **Test** lists predictions for every evaluated example, reports loss, and reports class accuracy for classifiers or token accuracy for sequence tasks. For the details of training and testing, see [Training and testing](TRAINING.md).

For source attribution and reproduction of the digit data, see [CNN data](CNN-DATA.md).
