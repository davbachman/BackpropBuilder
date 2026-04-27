# Save and Import Current State Design

## Goal

Add a first-class save/import workflow that lets a user download the current Backprop Builder workspace and later restore it. The saved state must include the computation graph and all parameter values, and it should restore the surrounding lab state that affects what the user sees.

The old session summary export is removed along with lesson-specific UI. The save/import project file is the only JSON export in the app.

## Approaches Considered

### Recommended: Versioned Project State File

Create a new JSON format with a file marker and schema version. The file stores the current graph, visualization graph, parameter baseline, trace state, selections, epoch/loss, and display toggles. Import validates the file before changing app state.

This is the best fit because it is explicit, debuggable, and has a single purpose: restoring a saved workspace.

### Graph-Only Export

Download only `GraphModel` and re-evaluate it on import. This is simpler, but it would lose epoch count, current trace, visualization state, and the initial parameter values needed to preserve the workspace baseline.

### Reuse Session Summary JSON

This option is no longer available because the session summary export has been removed.

## Saved Format

The save file will be JSON:

```json
{
  "kind": "backprop-builder-state",
  "version": 1,
  "savedAt": "2026-04-26T00:00:00.000Z",
  "state": {
    "graph": {},
    "visualizationGraph": {},
    "initialParameterValues": {},
    "selectedNodeIds": [],
    "selectedGroupId": null,
    "phase": "edit",
    "traceSteps": [],
    "traceIndex": 0,
    "epoch": 0,
    "currentLoss": null,
    "display": {
      "showMath": true,
      "showGradient": true,
      "showCode": false,
      "showVisualization": false
    }
  }
}
```

`GraphModel` is already serializable and includes nodes, edges, visual groups, learning rate, node parameter values, current node values, gradients, and caches. Trainable parameter values are therefore preserved both in `graph.nodes[].params.value` and in the session baseline map.

## UI

Add two buttons to the left action stack near reset/randomize:

- `Save state`: downloads `backprop-builder-state-YYYY-MM-DD.json`.
- `Import state`: opens a hidden JSON file input and restores the saved workspace.

Import should push the current workspace onto the undo stack before replacing it, stop playback, clear pending node placement, and show a concise error message if the file cannot be imported.

No separate session summary export remains.

## Validation and Error Handling

Import must reject files that are not JSON, have the wrong `kind`, use an unsupported schema version, omit required graph fields, contain malformed nodes or edges, or contain an invalid phase. Rejected imports must leave the current workspace unchanged.

Validation will be implemented with small TypeScript guard functions in the session module rather than a new runtime schema dependency.

Imported graph structure does not need to be optimization-ready. A user can import an incomplete graph and continue editing it; the existing graph validation panel will report model issues after import.

## Testing

Add domain tests for:

- Creating a save file preserves graph fields, groups, learning rate, parameter values, trace state, selections, epoch/loss, and display toggles.
- Parsing rejects malformed JSON, wrong kind, unsupported version, and minimally invalid graph shapes.
- Parsing a valid save file returns cloned tensor values rather than sharing object references.

Add app tests for:

- Save state button downloads a JSON project file.
- Import state restores a graph and visible state such as epoch/loss and graph formulas.
- Invalid import shows an error and leaves the current graph unchanged.
