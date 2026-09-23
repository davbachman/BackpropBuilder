# BackpropBuilder: teaching and exploration guide

Start this version with `npm install` and `npm run dev`, then open the displayed URL. All ten lessons calculate locally. No account, remote inference, or model download is required.

## A short classroom routine

1. Choose a preset from the gallery. Read its question and predict an outcome before running it.
2. Run the forward calculation. Select one input, neuron, token, attention cell, or output coordinate.
3. Open that component and explain the actual numbers. Opening or closing a module never trains the model.
4. Change one thing, run again, and compare. Edited inputs or parameters invalidate old results until another forward calculation.

Use the breadcrumbs and **Up one level** to navigate. Multiple neighboring modules can remain open. **Fit view** resets magnification; scroll the network diagram to pan after zooming. The original graph builder retains canvas pan/zoom and persistent module controls. **Ungroup** removes a grouping; closing it does not.

## Linear, neuron, and network lessons

The first two lessons begin with `x = 2`, `w = 3`, `b = 1`, and target `9`.

- **Forward** gives `3 × 2 + 1 = 7`.
- **Loss** gives `(7 − 9)² = 4`.
- **Backward** gives weight gradient `−8` and bias gradient `−4`.
- **Update parameters**, at learning rate `0.05`, gives `w = 3.4`, `b = 1.2`, prediction `8`, and loss `1`.

The new lessons use full squared error (and mean squared error for a dataset). Existing graph projects retain their previous conventions, including the builder's legacy half-squared-error operation. Exporting a lesson to the builder selects scalar MSE to preserve the lesson's full-square calculation.

In **Single neuron**, switch identity to ReLU and try a negative input. ReLU passes positive weighted sums and replaces negative sums with zero. Select the neuron to open its response graph, weighted contributions, bias, and arithmetic. Formulas, Python reference, and gradients are optional.

In **Small network**, the default input `[1, 2]` produces hidden activations `[2.5, 1.5]`, then output `1`. The network, layers, neurons, arithmetic, and plotted predictions all read the same parameter state.

In **Network playground**, the fixed two-dimensional rule is `class 1 if x₁ × x₂ > 0`, otherwise class 0. The positive class occupies the two same-sign diagonal regions. The default seed supplies 72 training and 24 validation examples. Filled points are training examples; open points are validation examples. The background shows the model's probability of class 1. Select a point to trace it, or a neuron to see its response curve and two-dimensional response map. Validation points can be inspected but cannot be used for a single-example parameter update.

**Train 20 epochs** explicitly runs all four phases on the training split, using full-batch gradients. The training and validation loss chart records both splits. Changing hidden widths or adding/removing a layer creates new initial parameters while preserving the data seed.

For a controlled comparison:

1. Keep the dataset seed and training budget fixed.
2. Train, then **Save comparison**.
3. **Reset parameters**, change one setting, and train the same number of epochs.
4. Compare validation losses; restore a run to inspect its parameters.

**Reset parameters** preserves the dataset. **Regenerate data** advances its seed, preserves the current parameters, and starts a new loss history. The parameter seed takes effect on reset; the first three presets use deliberately fixed, small initial values.

## Probabilities and attention

**Probabilities** turns scores into a distribution using stable softmax. Add the same constant to every score and check that the probabilities stay the same.

**One-query attention** exposes queries, keys, values, scaled dot products, normalized attention weights, and the weighted output. Select a score or attention-weight cell to see its dot product. Select an output coordinate to see its weighted sum.

The default weights are `[¼, ½, ¼]` and values are `[[2, 0], [0, 2], [2, 2]]`, producing `[1, 1.5]`. The V-only intervention changes the middle value to `[0, 4]`: output becomes `[1, 2.5]`, while Q, K, and attention weights remain fixed. Reset the intervention to restore the original values.

**Causal self-attention** marks future positions as excluded. Their normalized weights are exactly zero. Change a future token/input and inspect an earlier row: its causal output must stay unchanged. Attention weights are computed from the input; they are distinct from the learned Q/K/V projection matrices.

## Tokens, blocks, and the miniature decoder

**Tokens and embeddings** exposes vocabulary IDs, learned embedding lookup, learned absolute positions, and their sum. Selecting a token follows that position through the model.

**Transformer block** and **Complete miniature decoder** use the same actual decoder architecture: two pre-normalization blocks, model width 8, two attention heads of width 4, feedforward width 16 with ReLU, five vocabulary tokens, and a maximum context of 12 positions. Open attention heads, query rows, feedforward layers, and neurons. Residual additions remain visible, with actual selected-coordinate arithmetic.

Switch between the initial and trained checkpoints to compare fixed prompts. Parameter edits have stable named identities; reset edits to restore the selected checkpoint. Checkpoint evidence exposes the training configuration and loss history.

The model learns a synthetic **red → green → blue** cycle. It does not understand ordinary language. `<bos>` starts with red; `<eos>` is reserved and untrained. Absolute positions 10–11 are also untrained, even though the interface allows a 12-token context.

Generation is an explicit cycle:

1. **Calculate scores** for the current prompt.
2. **Form distribution**.
3. **Choose token** with greedy selection, or seeded sampling at positive temperature and a chosen top-k.
4. **Append token**, then repeat from scores for the extended prompt.

Temperature and top-k affect sampling; they do not change learned parameters or the fixed prompt's base logits. Appending changes the prompt and requires a new forward calculation.

## Save, share, and present

**Save experiment** (network lessons) or **Save state** (transformer lessons) downloads a JSON file containing parameters, inputs, seeds, module view, experiment settings, and saved comparisons. Share that file directly and import it into the corresponding lesson. Import reproduces calculations from the saved model. **Undo** restores earlier experiment states. Save before changing lessons or reloading the page; lesson state is not automatically stored across navigation.

**Hide controls** gives the calculation more space. Network **Presentation mode** disables parameter/training controls while retaining inspection. Hide the preset navigation for projection. Optional formulas, code, and gradients can be configured before saving the prepared state. Keyboard users can Tab to selectable neurons, cells, and data points and activate them with Enter or Space.

**Blank builder** preserves the original free-editing workspace. Network lessons can also export their current scalar graph with **Open in graph builder**. For classifier exports, the builder graph ends at its two output scores; the lesson supplies the final softmax and classifier loss. Existing builder save files remain importable in the builder.

## Reproduce checkpoints and validate

Use a current Node.js version supporting TypeScript type stripping and JSON import attributes (Node 22.18+ or Node 24):

```sh
node --experimental-strip-types scripts/train-decoder.ts
```

The script runs 360 seeded Adam updates at learning rate `0.01`, using seed 7 and a fixed synthetic corpus. It writes both the initial/trained bundle at `src/learning/decoder-checkpoints.json` and the portable trained checkpoint at `public/checkpoints/mini-decoder.json`. Training prefix lengths are 4, 7, and 10; held-out validation lengths are 5 and 8, across all three cycle phases. Live transformer training in the browser is not provided; browser forward execution, checkpoint comparison, parameter experiments, and generation are local.

```sh
npm test
npm run build
npm run lint
```

The tests include numeric fixtures, finite-difference derivatives, shared-parameter gradient accumulation, causal invariance, V-only interventions, an independent decoder forward reference, saved-state restoration, preserved graph behavior, and rendered lesson interactions. See `docs/VALIDATION.md` for the final verification record and remaining scope limits.
