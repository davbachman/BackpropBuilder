# BackpropBuilder: explore one model at every scale

Run `npm install` and `npm run dev`, then open the displayed URL. Models, datasets, training, and generation run locally in the browser.

## Explore the architecture

Every preset opens on the same editable graph canvas. There is no separate transformer demonstration or network editor. Groups organize the actual computation; opening them does not change parameters or run another model.

Layouts place parameters beside the operations they feed and pack parallel branches into compact rows to reduce crossings and empty space. Targets sit just below the input stack, with a clear wire lane beneath intervening blocks and the loss nearby. Rounded wires route around blocks. Moving a block updates its wire routes; your manual offsets remain part of the saved view. **Compact layout** reapplies the automatic arrangement at every level and fits the model in view without changing its calculations or learned weights. You can undo this arrangement with Cmd/Ctrl+Z.

Every nested block contains its own calculations and internal wires. Weights and biases stay inside their neuron; connections to other blocks use boundary ports. In continuous zoom, dragging a calculation stops at its enclosing frame and leaves the camera still. If an earlier layout scattered calculations outside a frame, that level returns to its connected arrangement. Opening a neuron fits its internal wires as well as its nodes into view.

- Scroll to zoom. Drag a block to move it; drag empty canvas to select blocks. Two-finger click-drag (or right-mouse drag) pans the canvas. **Zoom reveals detail** gradually fades each card to expose its real inner blocks and wires. A card is fully transparent as it fills the canvas; zooming out restores it. The geometry stays fixed throughout, and the breadcrumb follows the region under the camera. Turn this option off to use explicit open/close navigation.
- Double-click a block, or use its corner arrow, to open and focus it.
- Containers that only repeat one child's contents are skipped. The linear and single-ReLU presets go directly from one neuron card to their computation graphs. Opening a leaf block centers its calculations, and maximum zoom stops at a readable close-up.
- Follow **Model → block → MLP → layer** breadcrumbs; **Up one level** closes the current region.
- In a dense transformer layer, click a neuron dot to expose its input coordinates, weights, products, sum, bias, and activation. Choose a token row in the inspector. Editing a weight coordinate changes the shared matrix in the executable model.
- Click an operation to inspect its formula and tensor. Matrix heatmaps reveal individual coordinates; source tensors and learned parameters are editable.
- On narrower windows, **Build** and **Inspect** open the side panels. **Model controls** returns from a selected calculation to its dataset or prompt controls.

## Watch information flow

**Step** advances the recorded calculation. **Play** advances automatically; **Finish phase** shows the completed current phase. **Run forward** computes all predictions without changing weights. A model with a loss can then step through backpropagation and an update.

Wires remain uncluttered: color intensity represents mean absolute magnitude on a fixed logarithmic scale. Violet carries forward values; coral carries gradients backward. Light moving bands show the direction of the current calculation. Click a wire for its actual tensor value and gradient contribution. A shared parameter's node gradient adds contributions from all its uses; one wire shows only its own contribution. Values and gradients not reached by the trace remain hidden. Reduced-motion settings keep the bands still.

**Run one full training step** performs forward evaluation, loss, backward propagation, and an SGD update. **Run 10 training steps** repeats that cycle on the current example. The learning-rate slider controls update size.

## Build a neuron and reuse it

Start with **Blank builder**, or edit a preset. Pick operations from the palette, place them, and connect output handles to input handles. A new operation appears at the canvas point you click, preserving the positions of existing blocks and the current zoom and pan. Connecting, replacing, or deleting wires also preserves the arrangement at every level; the wire routes and computation update immediately. Use **Compact layout** when you want an automatic rearrangement using the current connections. For a neuron, connect inputs and weights to multiplication nodes, combine the products with a bias, and connect the sum to an activation.

Drag empty canvas to select calculations and group them. Drag an expanded group's title to move its contents together. Manual placements persist through zooming, execution, and saving. A group can contain other groups. Use **Copy block**, **Duplicate**, or Cmd/Ctrl+C and Cmd/Ctrl+V to reuse a neuron, layer, or attention mechanism. Copies keep incoming source connections and independent copied parameters; their outputs are exposed for wiring into the next layer. **Ungroup** removes a container without deleting its calculations. Deleting a group removes its calculations. Cmd/Ctrl+Z restores edits and navigation.

Tensor operations include matrix multiplication, embedding lookup, transpose, slicing, concatenation, softmax, causal masking, normalization, reshape, mean, convolution, and average pooling. Select an operation to edit relevant axes, shapes, or input counts. Missing inputs and incompatible shapes are shown before execution.

## Learn from included datasets

For a model you build yourself, add a **Dataset** node and choose its data from the dropdown on the card or in its inspector. The card labels its output ports: **x** and **y** for 1D datasets, or **x1**, **x2**, and **y** for 2D datasets. Feature ports feed the model inputs; **y** supplies the target to a Target or Loss node. Choose one example or a numeric batch in the Dataset panel; the original toy datasets contain 20 examples. Switching datasets keeps the target connection attached to **y** even when the number of features changes; connections to features no longer present are removed.

Regression and classification presets include deterministic training and held-out samples. Select a dataset or point from **Model controls**. The prediction curve or decision-boundary colors use the parameters of the canvas model. Click a point to load it into that model and follow its computation.

**Train 1 epoch** and **Train 25 epochs** use the training split, with full-batch gradients. Training and test losses are reported separately. Dataset changes preserve parameters. The simple linear/ReLU models accept one-dimensional regression data; the small MLP accepts two-dimensional regression data; the classifier includes nonlinear two-dimensional classification datasets.

## Attention and the miniature transformer

The attention presets expose Q, K, V, dot products, scaling, softmax, and weighted messages. Causal masks exclude future positions exactly. Opening a full block reveals normalization, multiple heads, residual additions, and the feed-forward MLP on the same canvas.

The complete decoder has two pre-normalization blocks, width 8, two heads, feed-forward width 16, and a five-token vocabulary. Its trained checkpoint learns **red → green → blue**. It is a small working transformer, not a natural-language model.

Edit the prompt using `<bos>`, `red`, `green`, `blue`, and `<eos>`. **Apply prompt** updates token IDs and positions together. **Generate next token** runs the current edited graph, chooses a token greedily or by sampling, and appends it. Temperature and top-k change sampling, not the model's logits. The context limit is 12 tokens; generation stops there or at `<eos>`. Prompt edits refresh the synthetic next-token training targets.

The **Trained** and **Initial** buttons reload the corresponding model preset. `<eos>` and absolute positions 10–11 were not trained in the supplied checkpoint. Ordinary language and arbitrary long prompts are outside its task.

## Small image classifier

The digit CNN uses 8×8 handwritten digits from the UCI optical-recognition dataset distributed with scikit-learn; this is not MNIST. Its graph exposes learned 3×3 filters, ReLU, average pooling, flattening, and a ten-class output. Use its image controls to inspect examples, filters, feature maps, and the input region responsible for a selected feature-map cell. Filter edits affect the actual predictions.

Choose **Restore trained weights** to reload the included checkpoint, including its convolution filters and digit classifier weights.

## Build a transformer or CNN from scratch

Open **Blank builder**. The inspector includes expandable CNN and transformer recipes with the exact shapes and connections for working models. All operations run in the same editable graph as the presets.

- Select a Weight or Bias, enter its shape, and choose **Initialize tensor**. Xavier suits dense and embedding matrices; He suits ReLU layers and convolution filters; ones and zeros initialize normalization scales and biases. Each new node has its own reproducible random seed. You can still enter or edit individual values.
- Operation help explains the input-port order and tensor shapes. Transpose accepts an empty axes field for its default permutation. Mean can average the whole tensor or retain dimensions. Reshape supports one `-1` dimension, such as `-1, 1`, to work across different batch sizes.
- Select calculations and group them, then edit the block's name and kind. A grouped Matrix product → Add bias → Activation exposes its individual neurons automatically. Duplicate and reconnect groups to build deeper networks and multiple attention heads.
- Select any Convolution to inspect its filters by output filter and input channel, edit shared weights, and inspect its actual input and feature maps. Convolution uses valid padding and stride 1; average pooling uses 2×2 patches and stride 2. Filter size and channel counts come from the tensors you connect.

## Dataset blocks and model testing

Every gallery preset and **File → Starter** gets its inputs and targets through a visible Dataset block. Inputs and targets connected to the dataset are aliases; choose data on the dataset itself. Model constants, such as attention's `1/√d`, remain ordinary fixed Input nodes.

The Dataset selector includes:

- Line, cubic and plane regression; threshold, circle, parabola, XOR and introductory neuron classification.
- **Handwritten digits:** 500 normalized UCI 8×8 images, with 400 training and 100 held-out examples. Image shape is `8×8×1`; targets are integer digit IDs.
- **Color cycle** and **Counting:** synthetic next-token datasets. Outputs are token IDs, position IDs and aligned next-token targets. Color cycle has 5 vocabulary entries; Counting has 7. Sequences have varying lengths and a maximum prompt context of 12 tokens.
- Small query/key/value attention experiments and four-class score examples for explaining softmax.

Select a Dataset block to choose an example or split, train for one or five epochs, and evaluate training and held-out data. Image and token datasets supply one complete tensor example at a time. Numeric datasets also support batches. Training shuffles training examples (or uses a numeric batch), yields to keep the interface responsive, and restores the example you were inspecting. **Stop training** discards that unfinished run. Held-out examples never update weights during epoch training. Evaluation reports loss and, for classifiers, class or token accuracy without changing parameters.

These controls work with user-created node names and connections. Connect raw logits and the dataset's target IDs to Cross-entropy; connect numeric predictions and targets to a regression loss. For a sequence model, prompt editing and generation use the dataset's vocabulary and the logits connected to its loss. A Softmax branch is useful for viewing probabilities, but Cross-entropy takes the raw logits.

## Save and reproduce

**File → Save** downloads the complete graph, parameters, inputs, group hierarchy, inspection view, and recorded execution state. **File → Import** restores it. Save before switching presets or reloading; those actions start a fresh preset. There is no hosted sharing service and no saved-project migration guarantee.

```sh
npm test
npm run build
npm run lint
npm run train:decoder
```

The decoder checkpoint script uses 360 seeded Adam updates and writes both the bundled checkpoint and `public/checkpoints/mini-decoder.json`. The browser supports inspection and SGD training of the same graph.
