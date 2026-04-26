# Edge Editing Design

## Context

Backprop Builder uses `@xyflow/react` for canvas interaction. `GraphCanvas.tsx` converts domain `GraphEdge` objects into React Flow edges and handles edge deletion and connection creation. Visual group inputs and outputs are derived from boundary edges in `visualGroupInterface`.

## Requirements

- Selecting a wire should show a visible selected state.
- Dragging a new wire into an already occupied input slot should replace the existing incoming edge for that slot.
- Replacing an edge must still reject cycles.
- Deleting a boundary wire that enters a collapsed group should not remove the group's visible input handle.

## Design

`BuilderEdge` will include React Flow's selected edge state in its class list, and `App.css` will style `.builder-edge.is-selected` with a clear stroke and halo.

`GraphCanvas` will compute the target input slot for a new connection. If a slot already has an incoming edge, the connection is valid as long as replacing that edge would not create a cycle. On connect, the old edge for that target slot is removed and the new edge is appended as a single graph change.

`visualGroupInterface` will expose group input handles from the group's internal input slots. Existing boundary edges keep their current handle order. When a boundary edge is deleted, the target node slot is still exposed as an unconnected group input handle, so the user can reconnect into the group. Dropping a wire onto a group input handle will resolve that visual handle back to the hidden internal node input before replacement validation runs.

Group outputs follow the same rule: a grouped node with no outgoing edge exposes a collapsed-group output handle, and dragging from that visual handle resolves back to the hidden internal source node.

## Testing

- Add a BuilderEdge test for the selected class.
- Add GraphCanvas tests for replacing an occupied input slot.
- Add grouping tests proving deleted boundary edges leave group input handles visible.
- Run the targeted tests, full Vitest suite, lint, and build.
