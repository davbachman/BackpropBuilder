# Training, testing, and reports

[Guide home](../README.md) · [Get started](USAGE.md) · [Datasets](DATASETS.md) · [Blocks](BLOCKS.md)

A trainable graph needs a prediction connected to the first **Loss** port and a target connected to the second. Parameters update only when you run a training action. **Run forward** and **Test → Run inference** evaluate without changing them.

## Watch one calculation at a time

Open **Train**. **Run forward** computes current outputs and loss. **Step** (⇧ Space) advances through the forward pass, loss, backward gradients, and update; after an epoch finishes, another Step starts the next cycle. **Play** advances automatically at the selected playback speed. **Finish phase** advances to the end of the current phase, and **Step inside** opens a grouped calculation when available. Select a block or wire, then open **Data** to read its value and gradient. A parameter node's gradient includes all contributions to that parameter; a selected wire shows that connection's contribution.

**Run one full training step** performs a full forward, backward, and SGD update on the current training example. **Randomize parameters** in Train gives trainable values a fresh starting point.

## Train for multiple epochs

Set **Epochs per run**, **Report loss every**, and **Examples per update** in **Train**, then click **Run epochs** (⇧ Return). One example per update is stochastic gradient descent; an intermediate batch size gives mini-batch gradient descent; the full training-set size gives batch gradient descent. Leave the batch-size field blank to follow the Dataset block's output mode (one example or the full numeric batch). **Reshuffle training examples each epoch** is on by default. Each training example appears once per epoch, and the last batch may be smaller. Tensor-shaped image and sequence examples currently run one at a time unless their graph has an explicit batch dimension; the Train tab explains when a larger batch is unavailable.

A run updates parameters using only the training split. **Stop training** ends an unfinished run. **Reporting** plots training and held-out loss at epoch 0 and at the chosen reporting interval. Both curves are evaluated using the same parameters; held-out evaluation never updates them. The current epoch and training loss also appear in Train. If you use the held-out curve to choose a batch size or learning rate, that split is serving as a validation set, not an independent final test set.

The **Learning rate** slider sets the SGD update size. If loss or weights grow rapidly, reduce it and use **Randomize parameters** to restart from sensible weights. Large input features can make a learning rate that worked on a small toy dataset unstable. For a custom CSV with widely different feature scales, consider scaling columns before import. A flat loss may indicate disconnected parameters, an unsuitable activation/loss combination, or a rate that is too low.

## Test without updating weights

Open **Test**, choose **Held-out test set** or **Training set**, and click **Run inference**. The **Reporting** tab shows each actual and predicted value and the aggregate loss. Classification models additionally show class accuracy; sequence models show token accuracy. You do not need an Argmax or Softmax block just to obtain these reports: classification evaluation interprets the model output. To inspect probabilities explicitly on the graph, add a Softmax branch; send raw logits directly into multiclass cross entropy.

The prediction table has pages for larger datasets. Testing does not change the model's parameters. You can still select a particular dataset example in **Details** to trace it on the canvas.

Python and notebook exports carry the chosen batch size, reshuffling setting, epoch count, and reporting interval into a `torch.utils.data.DataLoader` training loop. This gives students a concrete example of how PyTorch forms batches without adding a DataLoader block to the visual model. The notebook also plots training and held-out loss when Matplotlib is available.

## Understand Reporting

For a valid graph with a Loss block and one distinct input feature upstream of its prediction, Reporting plots target points and the prediction function. With two distinct features it can show a two-dimensional classification view, including multiclass decision regions. Several Input blocks may reuse the same Dataset column; they still count as one feature on the plot. Models with more complex inputs still show recorded losses and test metrics even if a simple input-output plot is unavailable. Predictions that blow far outside the target range may fall outside the plot; the loss and selected block values help diagnose that.

If a block is red or a run button is unavailable, select the block for its validation message in **Details**. Check all required ports, expression variables, tensor dimensions, and the loss choice.
