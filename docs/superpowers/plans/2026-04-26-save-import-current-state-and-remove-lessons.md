# Save Import Current State and Remove Lessons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add reloadable project save/import and remove lesson-specific UI and state from Backprop Builder.

**Architecture:** Put the project-state JSON format, cloning, and validation in `src/domain/session.ts`, reusing `cloneGraph` and tensor helpers. Keep App responsible for wiring browser download/upload actions and for restoring React state after a validated import. Remove lesson drawer/progress UI, lesson state, the Session card, and the old session summary export.

**Tech Stack:** React 19, TypeScript, Vite, Vitest, Testing Library, browser File/Blob APIs.

---

### Task 1: Domain Save File Format

**Files:**
- Modify: `src/domain/types.ts`
- Modify: `src/domain/session.ts`
- Test: `src/domain/session.test.ts`

- [x] **Step 1: Write failing domain tests**

Add `src/domain/session.test.ts` with tests for `createProjectStateFile` and `parseProjectStateFile`. Use `createStarterGraph`, `forwardPass`, and `scalarValue` to verify graph values, trace state, display toggles, and clone behavior.

- [x] **Step 2: Run domain tests to verify failure**

Run: `npm test -- src/domain/session.test.ts`

Expected: FAIL because `createProjectStateFile` and `parseProjectStateFile` are not exported.

- [x] **Step 3: Implement domain types and helpers**

Add `ProjectDisplayState`, `ProjectStateSnapshot`, `ProjectStateFile`, and `ProjectStateParseResult` to `src/domain/types.ts`. Implement `createProjectStateFile`, `parseProjectStateFile`, graph guards, tensor guards, trace guards, and clone helpers in `src/domain/session.ts`.

- [x] **Step 4: Run domain tests to verify pass**

Run: `npm test -- src/domain/session.test.ts`

Expected: PASS.

### Task 2: App Save and Import UI

**Files:**
- Modify: `src/App.tsx`
- Test: `src/App.test.tsx`

- [x] **Step 1: Write failing app tests**

Add tests that click `Save state`, inspect the downloaded JSON blob, import a valid project JSON through the hidden file input, and import invalid JSON to confirm the graph stays unchanged and an error message appears.

- [x] **Step 2: Run app tests to verify failure**

Run: `npm test -- src/App.test.tsx`

Expected: FAIL because save/import controls do not exist.

- [x] **Step 3: Implement app save/import wiring**

Import `Upload`, `createProjectStateFile`, `downloadProjectStateFile`, and `parseProjectStateFile`. Add a hidden JSON file input, `Save state` and `Import state` buttons, import error state, and restore logic that pushes undo history, stops playback, clears pending placement, and restores graph/display/session state from the parsed file.

- [x] **Step 4: Run app tests to verify pass**

Run: `npm test -- src/App.test.tsx`

Expected: PASS.

### Task 3: Remove Lesson UI and State

**Files:**
- Modify: `src/App.tsx`
- Modify: `src/App.test.tsx`
- Optional cleanup: `src/domain/types.ts`

- [x] **Step 1: Write failing removal tests**

Update the workspace render test so it expects no `Lesson drawer`, `Lesson progress`, `Session` card, or `Download session summary` button.

- [x] **Step 2: Run app tests to verify failure**

Run: `npm test -- src/App.test.tsx`

Expected: FAIL while lesson UI is still rendered.

- [x] **Step 3: Remove lesson and session summary state and UI**

Remove `GraduationCap` import, `lessons` import, `currentLesson`, `lessonOpen`, and `completedActions` state. Change `loadGraph` to take only a graph. Remove lesson drawer, lesson progress, the neutral Session section, summary export handler, and summary helper types/functions.

- [x] **Step 4: Run app tests to verify pass**

Run: `npm test -- src/App.test.tsx`

Expected: PASS.

### Task 4: Full Verification

**Files:**
- Modify only if verification reveals a defect.

- [x] **Step 1: Run full test suite**

Run: `npm test`

Expected: PASS.

- [x] **Step 2: Run lint**

Run: `npm run lint`

Expected: PASS.

- [x] **Step 3: Run production build**

Run: `npm run build`

Expected: PASS.
