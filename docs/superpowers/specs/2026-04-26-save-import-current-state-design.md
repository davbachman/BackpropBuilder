# Save and Import Current State Design

## Goal

Add a first-class save/import workflow that lets a user download the current Backprop Builder workspace and later restore it. The saved state must include the computation graph and all parameter values, and it should restore the surrounding lab state that affects what the user sees.

The existing session summary export remains separate. It is a report for lessons and LMS-style submission, not a reloadable project file.

## Approaches Considered

### Recommended: Versioned Project State File

Create a new JSON format with a file marker and schema version. The file stores the current graph, visualization graph, parameter baseline, trace state, lesson progress, selections, epoch/loss, and display toggles. Import validates the file before changing app state.

This is the best fit because it is explicit, debuggable, and does not overload the existing summary export.

### Graph-Only Export

Download only `GraphModel` and re-evaluate it on import. This is simpler, but it would lose lesson progress, epoch count, current trace, visualization state, and the initial parameter values used by the session summary.

### Reuse Session Summary JSON

Extend the current summary file so it can also be imported. This creates an ambiguous artifact: one JSON file would have to serve both as a final report and as a mutable project checkpoint. It also does not currently contain enough UI and trace state to restore a workspace accurately.

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
    "currentLessonId": null,
    "completedActions": [],
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

`currentLessonId` is stored instead of the full lesson definition because lesson objects contain functions. Import resolves the id through the built-in `lessons` list; missing ids restore as free exploration.

## UI

Add two buttons to the left action stack near reset/randomize:

- `Save state`: downloads `backprop-builder-state-YYYY-MM-DD.json`.
- `Import state`: opens a hidden JSON file input and restores the saved workspace.

Import should push the current workspace onto the undo stack before replacing it, stop playback, clear pending node placement, and show a concise error message if the file cannot be imported.

The existing `Download session summary` button remains in lesson progress.

## Validation and Error Handling

Import must reject files that are not JSON, have the wrong `kind`, use an unsupported schema version, omit required graph fields, contain malformed nodes or edges, or contain an invalid phase. Rejected imports must leave the current workspace unchanged.

Validation will be implemented with small TypeScript guard functions in the session module rather than a new runtime schema dependency.

Imported graph structure does not need to be optimization-ready. A user can import an incomplete graph and continue editing it; the existing graph validation panel will report model issues after import.

## Testing

Add domain tests for:

- Creating a save file preserves graph fields, groups, learning rate, parameter values, trace state, lesson id, and display toggles.
- Parsing rejects malformed JSON, wrong kind, unsupported version, and minimally invalid graph shapes.
- Parsing a valid save file returns cloned tensor values rather than sharing object references.

Add app tests for:

- Save state button downloads a JSON project file without replacing the session summary export.
- Import state restores a graph and visible state such as epoch/loss or selected lesson name.
- Invalid import shows an error and leaves the current graph unchanged.

