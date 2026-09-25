# Training, testing, and reports

[Guide home](../README.md) · [Get started](USAGE.md) · [Datasets](DATASETS.md) · [Blocks](BLOCKS.md)

A trainable graph needs a prediction connected to the first **Loss** port and a target connected to the second. Parameters update only when you run a training action. **Run forward** and **Test → Run inference** evaluate without changing them.

## Watch one calculation at a time

Open **Train**. **Run forward** computes current outputs and loss. **Step** (⇧ Space) advances through the forward pass, loss, backward gradients, and update; after an epoch finishes, another Step starts the next cycle. **Play** advances automatically at the selected playback speed. **Finish phase** advances to the end of the current phase, and **Step inside** opens a grouped calculation when available. Click a block or wire to read its value and gradient in **Details**. A parameter node's gradient includes all contributions to that parameter; a selected wire shows that connection's contribution.

**Run one full training step** performs a full forward, backward, and SGD update on the current training example. **Randomize parameters** in Train gives trainable values a fresh starting point.

## Train for multiple epochs

Set **Epochs per run** and **Report loss every** in **Train**, then click **Run epochs** (⇧ Return). A run trains on the training split only. **Stop training** ends an unfinished run. **Reporting** shows the recorded loss history at your chosen interval, after any applicable prediction visualization. The current epoch and loss also appear in Train.

The **Learning rate** slider sets the SGD update size. If loss or weights grow rapidly, reduce it and use **Randomize parameters** to restart from sensible weights. Large input features can make a learning rate that worked on a small toy dataset unstable. For a custom CSV with widely different feature scales, consider scaling columns before import. A flat loss may indicate disconnected parameters, an unsuitable activation/loss combination, or a rate that is too low.

## Test without updating weights

Open **Test**, choose **Held-out test set** or **Training set**, and click **Run inference**. The **Reporting** tab shows each actual and predicted value and the aggregate loss. Classification models additionally show class accuracy; sequence models show token accuracy. You do not need an Argmax or Softmax block just to obtain these reports: classification evaluation interprets the model output. To inspect probabilities explicitly on the graph, add a Softmax branch; send raw logits directly into multiclass cross entropy.

The prediction table has pages for larger datasets. Testing does not change the model's parameters. You can still select a particular dataset example in **Details** to trace it on the canvas.

## Understand Reporting

For a valid graph with a Loss block and one input upstream of its prediction, Reporting plots target points and the prediction function. With two inputs it can show a two-dimensional classification view. Models with more complex inputs still show recorded losses and test metrics even if a simple input-output plot is unavailable. Predictions that blow far outside the target range may fall outside the plot; the loss and selected block values help diagnose that.

If a block is red or a run button is unavailable, select the block for its validation message in **Details**. Check all required ports, expression variables, tensor dimensions, and the loss choice.
