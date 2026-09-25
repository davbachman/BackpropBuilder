# Canvas and navigation

[Guide home](../README.md) · [Get started](USAGE.md) · [Blocks](BLOCKS.md)

The canvas is one executable graph at every scale. A group is a view of its contained calculations, not a separate model with separate weights. The **Builder cards / Architecture cards** control changes presentation without changing computation.

## Add and connect

Select a block in **Build**, then click where it should appear. Or double-click blank canvas, type part of a block name, use the arrow keys if needed, and press **Enter** to place it at that point. **Escape** closes the menu. Adding a block or connecting a wire preserves the positions of blocks already on the canvas.

Drag from an output handle to an input handle. Ports on a block follow its input order; the [block reference](BLOCKS.md) lists the important ones. Click a wire to inspect its forward value and gradient contribution in **Details**. Violet carries values forward; coral shows gradients backward. Color strength reflects magnitude, and moving light bands show direction.

Drag a block to move it. Drag empty canvas to select several blocks. Two-finger click-drag or right-mouse drag pans; scroll to zoom. **Fit model** brings the graph into view. **Compact layout** reapplies an automatic arrangement when you want one; it does not change calculations or weights, and you can undo it with **Edit → Undo** or ⌘/Ctrl+Z.

## Group and reuse calculations

Select several blocks and group them. Groups can represent a neuron, a layer, a head of attention, or a whole block; groups can contain other groups. Right-click a group to rename it. Its name appears in the code outline. Copy and paste or duplicate a selection from **Edit** or with ⌘/Ctrl+C and ⌘/Ctrl+V. Copies have independent parameters. **Ungroup module** removes the container while retaining its calculations.

Double-click a group or its corner arrow to explore inside. **Zoom reveals detail** fades the group's card as it fills the view and exposes its real inner wires and blocks. Turn it off for explicit open/close navigation. The breadcrumb and **Up one level** move between regions. Very simple groups may skip redundant intermediate cards.

## Navigate with code

Open **Code** in the right sidebar. Each block has a pseudocode line; a group appears as a collapsible function call such as `y = func(x)` with its inner operations indented below. Click a line to center and select the corresponding canvas block. Selecting a block on the canvas highlights and scrolls to its code line. Click a block's name at the top of **Details** to rename it; press **Enter** or leave the field to save, or **Escape** to cancel.

The code outline is a learning and navigation view. For runnable Python, use [PyTorch export](FILES-AND-EXPORT.md).
