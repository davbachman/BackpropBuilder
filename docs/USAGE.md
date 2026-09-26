# Get started

[Guide home](../README.md) · [Canvas](CANVAS.md) · [Blocks](BLOCKS.md) · [Datasets](DATASETS.md) · [Training](TRAINING.md)

Open the [app](https://davbachman.github.io/BackpropBuilder/). It starts on a blank builder canvas so a class can construct a model from its first calculation. You can also [download an example project](../public/models/README.md) and choose **File → Import**.

## Find your way around

| Area | Use it for |
| --- | --- |
| **Build** in the left sidebar | Choose a block, then click the canvas to place it. Double-click empty canvas for a searchable block menu. |
| **Train** in the left sidebar | Run forward, step through a calculation, play a trace, or train for a chosen number of epochs. |
| **Test** in the left sidebar | Run inference on held-out or training examples without changing parameters. |
| **Canvas** | Connect ports with wires, move and group blocks, and zoom into a group's calculations. |
| **Details** in the right sidebar | Edit a selected block and inspect values, gradients, and errors. |
| **Code** in the right sidebar | Browse an expandable pseudocode outline and navigate to the corresponding blocks. |
| **Reporting** in the right sidebar | See an applicable graph, recorded losses, and test predictions or accuracy. |

Both sidebars can be resized or collapsed. The **Dark** button in the top bar switches between light and dark themes; the browser remembers your choice. The clickable **Backprop Builder** title opens **About** and **Reference** (the online guide). **File** contains New, Save, Import, and PyTorch exports; **Edit** contains Undo, Copy, Paste, and Duplicate. **Randomize parameters** is in **Train**.

## Build your first regression model

For the quickest guided start, import [`linear.json`](../public/models/linear.json) and inspect its Dataset → Input → Param and arithmetic → Loss path. To assemble the same idea yourself:

1. Add a **Dataset** block and choose **Line regression (1D)**. Add an **Input** and connect the dataset's feature output to it.
2. Add two **Param** blocks for a slope and intercept. Add an **Arithmetic** block and enter `x1 * x2`; connect the Input and slope to its ports.
3. Add another **Arithmetic** block and enter `x1 + x2`; connect the product and intercept. Add a **Target** block and connect the dataset's target output to it.
4. Add a **Loss** block, select **Mean squared error**, connect the model's prediction to the first loss port and the Target to the second. Select blocks to inspect or rename them in **Details**.
5. Open **Train**. Use **Run forward** to inspect predictions, **Step** to see each forward and backward calculation, or set **Epochs per run** and choose **Run epochs**. Open **Reporting** to see the fit and loss history.

An arithmetic block's expression uses its port names `x1`, `x2`, and so on. All required ports must be wired before the graph can run. If a block has a problem, it is highlighted red; select it for the message in **Details**.

After training, open **Test**, choose **Held-out test set**, and select **Run inference**. The **Reporting** tab then shows predictions and the held-out loss. Classification models also show accuracy.

## Save your work

Choose **File → Save** to download an editable project JSON file. **File → Import** opens that file later; **File → New** returns to a blank canvas. Save your current work before importing another project. See [Saving and exporting](FILES-AND-EXPORT.md) for Python and notebook exports.
