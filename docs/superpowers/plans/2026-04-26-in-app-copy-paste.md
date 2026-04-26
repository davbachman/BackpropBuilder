# In-App Copy/Paste Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add in-memory copy/paste for selected graph nodes and selected visual groups.

**Architecture:** Put graph-fragment copy and paste logic in a tested domain helper, then wire keyboard shortcuts in `App.tsx`. App state will hold the in-memory clipboard fragment and paste count; the domain helper will handle cloning, generated id/label remapping, selected-node edge filtering, group boundary preservation, and position offsets.

**Tech Stack:** React 19, TypeScript, Vite, Vitest, Testing Library, `@xyflow/react`.

---

## File Structure

- Create `src/domain/clipboard.ts`: copy/paste helper types and pure functions.
- Create `src/domain/clipboard.test.ts`: unit tests for node and group copy/paste behavior.
- Modify `src/App.tsx`: in-memory clipboard state and keyboard handlers.
- Modify `src/App.test.tsx`: integration coverage for keyboard copy, paste, and undo.

### Task 1: Domain Clipboard Helper

**Files:**
- Create: `src/domain/clipboard.ts`
- Test: `src/domain/clipboard.test.ts`

- [ ] **Step 1: Write the failing domain tests**

```ts
import { describe, expect, it } from 'vitest'
import { createStarterGraph } from './examples'
import { copyGraphSelection, pasteGraphClipboard } from './clipboard'
import { mergeNodesIntoVisualGroup } from './grouping'

describe('graph clipboard', () => {
  it('copies selected nodes and pastes them with remapped ids and internal edges only', () => {
    const graph = createStarterGraph()

    const fragment = copyGraphSelection(graph, { nodeIds: ['x', 'w', 'mul'] })
    const result = pasteGraphClipboard(graph, fragment!, { x: 40, y: 50 })

    expect(fragment).toBeDefined()
    expect(result.selection.groupId).toBeUndefined()
    expect(result.selection.nodeIds).toHaveLength(3)
    expect(result.graph.nodes).toHaveLength(graph.nodes.length + 3)
    expect(result.graph.edges).toHaveLength(graph.edges.length + 2)

    const pastedNodes = result.selection.nodeIds.map((nodeId) => result.graph.nodes.find((node) => node.id === nodeId))
    expect(pastedNodes.map((node) => node?.label)).toEqual(['x', 'w', 'x * w'])
    expect(pastedNodes.map((node) => node?.position)).toEqual([
      { x: 80, y: 110 },
      { x: 80, y: 310 },
      { x: 360, y: 210 },
    ])

    const pastedNodeIds = new Set(result.selection.nodeIds)
    const pastedEdges = result.graph.edges.filter((edge) => pastedNodeIds.has(edge.source) && pastedNodeIds.has(edge.target))
    expect(pastedEdges).toHaveLength(2)
    expect(pastedEdges.map((edge) => edge.inputSlot)).toEqual([0, 1])
    expect(result.graph.edges.some((edge) => pastedNodeIds.has(edge.source) && edge.target === 'add')).toBe(false)
  })

  it('copies a selected visual group and pastes it as a new collapsed group', () => {
    const groupedGraph = mergeNodesIntoVisualGroup(createStarterGraph(), ['mul', 'add']).graph

    const fragment = copyGraphSelection(groupedGraph, { nodeIds: [], groupId: 'group-1' })
    const result = pasteGraphClipboard(groupedGraph, fragment!, { x: 25, y: 35 })

    expect(fragment).toBeDefined()
    expect(result.selection.nodeIds).toEqual([])
    expect(result.selection.groupId).toBeDefined()
    expect(result.graph.nodes).toHaveLength(groupedGraph.nodes.length + 2)
    expect(result.graph.edges).toHaveLength(groupedGraph.edges.length + 1)
    expect(result.graph.groups).toHaveLength(2)

    const pastedGroup = result.graph.groups?.find((group) => group.id === result.selection.groupId)
    expect(pastedGroup).toBeDefined()
    expect(pastedGroup?.label).toBe('Group 1')
    expect(pastedGroup?.nodeIds).toHaveLength(2)
    expect(pastedGroup?.position).toEqual({
      x: groupedGraph.groups![0].position.x + 25,
      y: groupedGraph.groups![0].position.y + 35,
    })

    const pastedGroupNodeIds = new Set(pastedGroup?.nodeIds)
    const pastedInternalEdges = result.graph.edges.filter(
      (edge) => pastedGroupNodeIds.has(edge.source) && pastedGroupNodeIds.has(edge.target),
    )
    expect(pastedInternalEdges).toHaveLength(1)
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- src/domain/clipboard.test.ts`

Expected: FAIL because `src/domain/clipboard.ts` and its exports do not exist.

- [ ] **Step 3: Implement minimal domain helper**

Create `GraphClipboardFragment`, `GraphClipboardSelection`, `copyGraphSelection`, and `pasteGraphClipboard`. Use `cloneGraph` to avoid sharing nested tensor/cache objects, filter internal edges with a selected-id set, remap node ids through a unique-id helper, remap internal edges, and remap the copied group when present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test -- src/domain/clipboard.test.ts`

Expected: PASS with both clipboard tests passing.

- [ ] **Step 5: Commit**

```bash
git add src/domain/clipboard.ts src/domain/clipboard.test.ts
git commit -m "feat: add graph clipboard helper"
```

### Task 2: App Keyboard Integration

**Files:**
- Modify: `src/App.tsx`
- Test: `src/App.test.tsx`

- [ ] **Step 1: Write the failing app test**

Add this test to `src/App.test.tsx` near the existing undo tests:

```ts
it('copies and pastes the selected graph node with Command-C and Command-V', async () => {
  const user = userEvent.setup()
  render(<App />)

  await user.click(screen.getByRole('button', { name: /Load starter example/i }))

  fireEvent.keyDown(document, { key: 'c', metaKey: true })
  fireEvent.keyDown(document, { key: 'v', metaKey: true })

  expect(screen.getByText('9 nodes, 7 edges')).toBeInTheDocument()

  fireEvent.keyDown(document, { key: 'z', metaKey: true })

  expect(screen.getByText('8 nodes, 7 edges')).toBeInTheDocument()
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- src/App.test.tsx -t "copies and pastes the selected graph node"`

Expected: FAIL because the app does not yet handle `Cmd/Ctrl+C` or `Cmd/Ctrl+V`.

- [ ] **Step 3: Wire app clipboard state and keyboard shortcuts**

Import the helper from `src/domain/clipboard.ts`. Add `clipboard` and `clipboardPasteCount` state to `App.tsx`. Extend the document keydown effect to ignore editable targets, preserve undo behavior, copy the current selection on `Cmd/Ctrl+C`, and paste with offset `{ x: 36 * (clipboardPasteCount + 1), y: 36 * (clipboardPasteCount + 1) }` on `Cmd/Ctrl+V`.

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- src/App.test.tsx -t "copies and pastes the selected graph node"`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/App.tsx src/App.test.tsx
git commit -m "feat: add graph copy paste shortcuts"
```

### Task 3: Full Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run full tests**

Run: `npm test`

Expected: PASS.

- [ ] **Step 2: Run lint**

Run: `npm run lint`

Expected: exit 0.

- [ ] **Step 3: Run production build**

Run: `npm run build`

Expected: exit 0.

- [ ] **Step 4: Inspect final diff**

Run: `git status --short && git diff --stat HEAD`

Expected: only intended feature changes remain uncommitted, or no changes remain if all task commits have been made.
