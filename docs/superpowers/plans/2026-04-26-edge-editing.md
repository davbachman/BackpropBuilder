# Edge Editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make edge selection visible, allow input-slot replacement, and keep collapsed group inputs visible after boundary edge deletion.

**Architecture:** Keep visual styling in `BuilderEdge` and `App.css`. Keep connection replacement in `GraphCanvas` so React Flow and domain graph stay consistent. Extend `visualGroupInterface` to return input handles for exposed internal target slots, with optional `edgeId` for connected boundary handles.

**Tech Stack:** React 19, TypeScript, Vite, Vitest, Testing Library, `@xyflow/react`.

---

## Tasks

### Task 1: Selected Edge Styling

**Files:**
- Modify: `src/components/BuilderEdge.tsx`
- Modify: `src/App.css`
- Test: `src/App.test.tsx`

- [ ] Write a failing test that renders `BuilderEdge` with `selected: true` and expects `is-selected` on the edge path.
- [ ] Run `npm test -- src/App.test.tsx -t "selected wire"`.
- [ ] Add selected class support and CSS selected styling.
- [ ] Re-run the targeted test.

### Task 2: Replace Occupied Input Slot

**Files:**
- Modify: `src/components/GraphCanvas.tsx`
- Test: `src/App.test.tsx`

- [ ] Write a failing GraphCanvas test that connects into an occupied input slot and expects the old slot edge removed and the new edge added.
- [ ] Run the targeted test and confirm it fails.
- [ ] Adjust connection validation to allow replacement and run cycle checks against the graph with the replaced edge removed.
- [ ] Adjust connect handling to remove the replaced edge and add the new edge in one `onGraphChange` call.
- [ ] Re-run the targeted test.

### Task 3: Stable Group Input Handles

**Files:**
- Modify: `src/domain/grouping.ts`
- Test: `src/domain/grouping.test.ts`
- Test: `src/App.test.tsx`

- [ ] Write a failing grouping test that deletes a boundary edge entering a group and expects the input handle count to stay exposed.
- [ ] Run `npm test -- src/domain/grouping.test.ts`.
- [ ] Extend `visualGroupInterface` so input handles can exist without an `edgeId`.
- [ ] Update `GraphCanvas` to use optional `edgeId` for handle lookup and group metrics.
- [ ] Re-run grouping and app tests.

### Task 4: Verification

**Files:**
- Verify all changed files.

- [ ] Run `npm test`.
- [ ] Run `npm run lint`.
- [ ] Run `npm run build`.
- [ ] Commit the completed fix.
