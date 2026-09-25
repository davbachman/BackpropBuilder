# Validation and scope

The app opens on a blank canvas. Fourteen public, importable model files run through the editable `GraphModel` execution API. The architecture view, scalar arithmetic inspection, parameter editor, dataset experiments, and token generation share canonical parameters.

## Verification

Run `npm test`, `npm run build`, and `npm run lint` from the project root.

The latest test run on September 24, 2026 passed **441 tests across 39 files**.

The suite checks:

- Every public example imports as an editable, dataset-backed project and executes in the same graph workspace.
- Transformer logits, probabilities, and intermediate tensors match the reference decoder for both checkpoints.
- All transformer parameter gradients agree with reference autograd; an SGD update reduces loss.
- Finite-difference checks cover broadcasting, embedding reuse, matrix operations, softmax, masking, normalization, cross-entropy, convolution, and pooling.
- Dense-neuron projections do not mutate canonical graphs. Edited scalar coordinates map to shared tensor parameters. Wire gradients are per-use contributions; node gradients accumulate shared uses.
- Forward and backward traces hide unreached results. Flow bands reverse direction for backpropagation and respect reduced motion.
- Nested groups preserve values, gradients, selection, copying, undo, and saved inspection state. Recursive layouts keep siblings from overlapping.
- Every preset is checked at each hierarchy level for wires crossing unrelated blocks. Routing also covers projected neuron arithmetic, staggered obstacles, live geometry changes, and forward/backward animation along the same rounded path.
- Continuous zoom keeps every nested region in fixed canvas coordinates, fades covers in both directions, and joins forward/gradient wire segments at exact card boundary ports. Tests cover all eleven presets, nested containment, routing clearance, subpixel geometry, camera stability, and dragging revealed regions.
- Redundant single-child containers are skipped without altering the executable model. Linear/ReLU neurons reveal their arithmetic directly; navigation targets calculation bounds, and operations and wires remain accessible at maximum zoom.
- All presets keep nested wire routes inside their owning frames. Regressions cover misplaced neuron weights/biases, bounded routing around obstacles, one-time layout repair, drag boundaries, and camera stability while arranging calculations.
- Linear and ReLU overviews keep targets close to their inputs and align target-to-loss wires beneath the neuron without a large empty band. The Compact layout action clears manual placements and fits the scene without changing the computation graph or trained parameters.
- Repeated palette insertions across every preset preserve existing block geometry and place new nodes at the clicked canvas coordinates. Tests cover both zoom modes, manual offsets, projected neuron inspection, camera stability, and undo.
- Connection edits preserve layout at every hierarchy level across all presets and both zoom modes. Tests verify live execution/routing, projected neurons, further insertions, camera stability, Compact layout, undo, and saved layout isolation.
- Safari was checked at neuron computation depth through forward and backward steps and further zooming: text and wires remain sharp. Compositing stays on the unscaled canvas container; flow bands retain their own forward/backward animation. Browser checks also confirmed unchanged node bounds and camera position during stepping.
- Native browser text selection is disabled on the canvas, with editable fields exempted. The reported Safari selection highlight cleared after this change; actual wire creation and canvas box selection were checked independently.
- Block dragging preserves manual offsets through nested zoom and saved projects. Expanded groups move their contents together, primary drags select/move blocks, and secondary pointer drags pan without changing selections or model parameters.
- Copied neuron outputs remain available for wiring without duplicating occupied destination ports.
- Dataset plots and training read the current canvas parameters; training/test splits remain separate.
- Dataset nodes expose the regression, classification, image, sequence, attention, and score datasets on the card and in the inspector, with labeled feature and target ports. Both zoom modes preserve feature/target values and target wiring through 1D/2D changes and undo.
- A named custom group renders as an expandable function call in Code view. Right-clicking a group opens its rename control. Loss selectors and side ports remain the same in builder and architecture presentations.
- Decoder generation uses live edited parameters, validates prompts, refreshes positions, and stops at the context or end-token limit.

## Decoder checkpoint evidence

The decoder has two pre-normalized blocks, model width 8, two attention heads of width 4, ReLU feed-forward width 16, learned positions for 12 tokens, and a five-token vocabulary. Input embeddings and output projection are separate parameters.

It learns the repeating sequence `red → green → blue`. Training prefix lengths are 4, 7, and 10; validation lengths are 5 and 8, across all three cycle phases. `<eos>` is reserved and untrained.

The supplied checkpoint records training loss `0.0005509947040016146` and validation loss `0.000549696741021241` after 360 Adam updates with learning rate 0.01 and seed 7. The automated suite reproduces its parameters. These metrics describe only the synthetic task.

## Practical limits

- The transformer is a complete tiny decoder trained on a color cycle. It does not understand ordinary language. Its context is limited to 12 tokens.
- Computation and bounded training run on the browser's main thread. Large models and datasets are not the target.
- Opening tensor MLP arithmetic projects one neuron's coordinates for a selected token onto the canvas. Its parameters and gradients belong to the shared model; execution still uses the efficient tensor operations. User-built scalar neurons remain ordinary executable operation groups.
- Automatic layout supplies initial positions; manual offsets persist across semantic views. Graph edits and connections change the executable topology. Groups are views, not separate numerical modules.
- Save/import uses local JSON files. Examples are downloadable from `public/models`; opening a file replaces the current graph, so save work you want to retain. Broad migration compatibility is intentionally outside scope.
- The digit example uses UCI 8×8 images rather than MNIST. Checkpoint metrics and dataset attribution are documented with the CNN assets.
- Keyboard controls, numeric alternatives to colors, responsive panels, and reduced-motion styles are provided. A formal screen-reader audit and physical touch-device test have not been performed.


## Scratch authoring and dataset-backed presets

- Palette-based integration tests assemble a two-filter CNN and a complete two-head transformer block from `createNode`, tensor initialization, and `connectGraphNodes`, without presets, checkpoints, or reserved node IDs. Both differentiate, reduce training loss, evaluate held-out data, and survive a project save/import round trip.
- Every public example is checked for a visible Dataset source and an executable graph; example changes preserve parameters. Dataset tests check aligned shapes, finite values, and train/test splits. Changing held-out labels leaves training updates unchanged; cancellation leaves the input graph unchanged.
- Numeric batch tests cover inferred reshape dimensions and switching between training, test and full batches. UI tests cover tensor initialization, operation settings, generic filter editing, prompt generation, and dataset training/evaluation.
- Browser verification: initialized a 4×3×3×1 He filter from Blank builder; evaluated the CNN (100% training / 97% held-out accuracy for the included checkpoint) and completed an epoch through the Dataset panel. The user’s Safari model was left intact.
