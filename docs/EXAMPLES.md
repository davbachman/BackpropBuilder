# Example models and lessons

[Guide home](../README.md) · [Get started](USAGE.md) · [Canvas](CANVAS.md) · [Datasets](DATASETS.md)

Backprop Builder opens blank. To use an example, download its JSON file from the [model directory](../public/models/README.md) and choose **File → Import**. These are editable projects on the same canvas and use visible Dataset blocks, so you can change their data, calculations, parameters, groups, and layouts.

| Start with | Files | What to explore |
| --- | --- | --- |
| A neuron or regression line | [Starter neuron](../public/models/starter-neuron.json), [linear model](../public/models/linear.json), [ReLU neuron](../public/models/neuron.json) | Trace a scalar forward pass, its loss, and each gradient. |
| Small classification | [Small network](../public/models/small-network.json), [2D playground](../public/models/playground.json) | Copy a neuron, form a layer, and watch a decision boundary change. |
| Probabilities and tokens | [Class scores](../public/models/probabilities.json), [embeddings](../public/models/embeddings.json) | Explore softmax, embedding tables, and token positions. |
| Attention | [One-query attention](../public/models/attention.json), [causal attention](../public/models/causal.json) | Follow query/key/value scores, the mask, probabilities, and weighted values. |
| Transformer | [Block](../public/models/block.json), [untrained block](../public/models/block-untrained.json), [decoder](../public/models/decoder.json), [untrained decoder](../public/models/decoder-untrained.json) | Zoom from a model into blocks, heads, MLPs, and individual calculations. Compare starting and trained weights. |
| Image classification | [Digit CNN](../public/models/cnn.json) | Inspect 3×3 filters, ReLU, feature maps, pooling, and ten-class predictions. |

The miniature decoder has two pre-normalization transformer blocks, width 8, two attention heads, and a feed-forward width of 16. Its synthetic training task is the **red → green → blue** cycle. It is an executable explanation of a transformer, not a general language model. Its prompt controls use the dataset's small vocabulary; generation uses the current edited graph.

The CNN classifies normalized 8×8 UCI handwritten digits, not 28×28 MNIST images. Select a sample and inspect its filters and feature maps in **Details**. The included trained checkpoint can be restored from the CNN controls. See [data attribution](CNN-DATA.md).

An effective class sequence is to trace the starter neuron, build the line model from scratch, copy neurons into an MLP, inspect attention's weighted sum, and finally unpack the decoder. The [block reference](BLOCKS.md) lists the operations needed to build transformer and CNN graphs yourself.
