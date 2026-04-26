# In-App Copy/Paste For Selected Nodes And Groups

## Context

Backprop Builder is a Vite/React graph editor built on `@xyflow/react`. The computational graph lives in `GraphModel`, while visual groups are metadata in `GraphModel.groups`. `App.tsx` owns selection, undo snapshots, phase/trace state, and graph updates. `GraphCanvas.tsx` maps graph nodes and groups to React Flow nodes and reports selection changes back to `App.tsx`.

## Requirements

- Copy selected computation nodes with `Cmd/Ctrl+C`.
- Paste the copied fragment with `Cmd/Ctrl+V`.
- Keep the clipboard in app memory only; do not read from or write to the OS clipboard.
- When copying multiple selected nodes, include only edges where both endpoints are selected.
- When copying a selected visual group, include its member nodes, internal edges, and group metadata so paste recreates the collapsed group.
- Pasted nodes, edges, and groups must receive ids that do not collide with the existing graph.
- Paste should offset node and group positions so the duplicate is visible.
- Paste should select the newly pasted nodes or pasted group.
- Paste should reset the editor back to edit mode, clear trace state and pending placement, and create one undo snapshot.
- Keyboard shortcuts must ignore text inputs, textareas, selects, and contenteditable elements.

## Design

Add a focused domain helper in `src/domain/clipboard.ts` that owns graph-fragment copy and paste behavior. `copyGraphSelection(graph, selection)` returns an in-memory clipboard fragment or `undefined` if the selection cannot be copied. `pasteGraphClipboard(graph, fragment, offset)` returns the next graph plus the selection that should become active after paste.

The clipboard fragment stores cloned nodes, cloned internal edges, and optionally a cloned group. Copying selected nodes stores no group. Copying a selected group stores that group and uses the group node ids as the fragment's selected nodes. Boundary edges that connect the group to outside nodes are not copied, because pasting a self-contained fragment should not attach itself to unrelated existing nodes.

Pasting creates a node-id map from original ids to unique ids. Edges and group node ids are remapped through that map. Node and group positions are translated by the supplied paste offset. App state tracks the copied fragment and a paste count; each paste of the same fragment uses a larger offset so repeated pastes do not overlap each other.

`App.tsx` extends its document-level keydown handler. `Cmd/Ctrl+C` copies the current node selection or selected group into React state and resets paste count. `Cmd/Ctrl+V` pushes history, pastes the fragment, selects the new nodes or group, resets phase/trace state to edit, clears pending placement, and increments paste count. Existing `Cmd/Ctrl+Z` behavior stays unchanged.

## Testing

- Add domain tests for copying and pasting selected nodes with only internal edges.
- Add domain tests for copying and pasting a selected visual group as a new collapsed group.
- Add an app-level keyboard test that copies a selected node, pastes it, and verifies undo removes the pasted node.
- Run the targeted new tests, the full Vitest suite, lint, and build before completion.
