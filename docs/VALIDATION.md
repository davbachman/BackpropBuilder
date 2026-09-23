# Validation and scope

The implementation has an executable miniature decoder, rather than a static transformer diagram. All computation is local. New network, probability, embedding, attention, and decoder lessons share the tensor/autodiff implementation in `src/learning/math.ts`. The existing free graph builder remains available with its own scalar/tensor graph execution API.

## Recorded checks

Run these commands from the project root:

```sh
npm test
npm run build
npm run lint
```

Final verification on 2026-09-23: **176 tests across 18 files passed**, the TypeScript/Vite production build passed, and ESLint passed. The production app is split into smaller lazy-loaded bundles; no bundle-size warning remains.

A real Chromium browser opened all ten presets, completed the exact neuron update fixture, trained the classifier, selected a neuron, checked the V-only attention intervention, traced a transformer token through an attention row and feedforward neuron, and completed scores → distribution → choose → append → recalculate. Browser console: zero errors and zero warnings. Screenshots were visually inspected at 1440×1000 and 820×1180; the gallery was also checked at 1280×800. Tablet decoder and free-builder layouts were checked, and the decoder page width matched its 820px viewport with no horizontal page overflow.

Local screenshots are in `output/playwright/`: `gallery-desktop.png`, `gallery-tablet.png`, `network-desktop.png`, `attention-head-desktop.png`, `decoder-tablet.png`, and `builder-tablet.png`. The automated rendered tests use React Testing Library and jsdom; the browser checks complement those tests.

| Requirement | Verified evidence |
| --- | --- |
| Persistent modules | `src/domain/nestedModules.test.ts` checks independent expansion, preserved values/gradients/identities/layout, hierarchy, explicit ungrouping, descendant movement, nested copying, and saved view state. |
| Rendered builder navigation | `src/App.modules.test.tsx` opens nested arithmetic while its neighbor stays collapsed, restores a selected internal node after closing/reopening, undoes navigation, and runs target-free inference. |
| Preserved builder editing | Existing App/domain suites cover node values, tensor values, datasets and ports, connections, copy/paste, graph edits, phase traces, saving/importing, and plots. |
| Elementwise versus matrix multiplication | `src/domain/matmul.test.ts` verifies non-square matrix multiplication and finite-difference gradients, including a parameter feeding both operands. Existing multiplication tests retain elementwise behavior. |
| Full-square neuron fixture | `src/learning/network.test.ts` verifies prediction 7, loss 4, gradients −8/−4, simultaneous update to weights 3.4/1.2, prediction 8, and loss 1. |
| Network calculation consistency | Every inspected scalar neuron reconstructs the batched tensor calculation. Finite differences check network weights/biases and classifier cross-entropy. Builder export is checked against the lesson calculation. |
| Controlled network training | Deterministic 2D labels and disjoint training/validation splits are verified. Repeated seeded training produces the same parameters; validation samples are excluded from updates. |
| Preset gallery and navigation | `src/learning/LearningStudio.test.tsx` launches all ten real lazy-loaded lessons, opens blank/prepared builders, and checks sidebar/gallery navigation. |
| Rendered transformer activities | `src/learning/TransformerLesson.test.tsx` checks attention masking/interventions, token/head/neuron tracing, selection and breadcrumb preservation, stale-result invalidation, distinct generation phases, parameter edits, undo, comparisons, and import. Saving a chosen sampled token preserves its advanced seed and reproduces the next generation cycle. |
| Rendered network activities | `src/learning/NetworkLesson.test.tsx` checks separate forward/loss/backward/update phases, independent modules, arithmetic navigation, stale-result invalidation, plotted-point selection, held-out protection, fixed training budgets, resets, and imports. |
| Tensor backward rules | `src/learning/math.test.ts` checks broadcasting reduction, matrix multiplication, shared paths, reductions, reshape, transpose, head slicing/concatenation, repeated embedding indices, softmax, masking, ReLU, cross-entropy, and layer normalization. |
| Independent transformer reference | `src/learning/decoder.test.ts` compares both supplied checkpoints with an independent ordinary-array implementation at each block, attention head, logit, and probability. |
| Attention intervention | The requested weights `[1/4,1/2,1/4]` produce `[1,1.5]`, and changing the middle value to `[0,4]` produces `[1,2.5]`. Decoder-level V interventions preserve the selected head's Q, K, and weights. |
| Causality | Changing later token IDs leaves earlier decoder outputs unchanged; every masked future attention entry is exactly zero. |
| Transformer gradients | Central finite differences check representative coordinates across embeddings, attention projections, normalization, both feedforward matrices, and output projection; repeated token IDs test shared embedding accumulation. |
| Inference and generation immutability | Decoder forward passes and seeded sampling leave parameters unchanged. Training changes parameters. Temperature/top-k affect sampling distributions; greedy selection works independently of positive-temperature sampling. |
| Reproducible checkpoints | The test retrains from seed 7 for the complete 360-update budget and compares every parameter exactly with the included trained checkpoint. |
| Saved experiments | Graph, network, and token-exploration tests restore inputs, parameters, seeds, display controls, and module selections, then reproduce numerical outputs. Invalid shapes and malformed files are rejected. |

## Checkpoint evidence

The decoder has two pre-normalized blocks, model width 8, two attention heads of width 4, ReLU feedforward width 16, learned positions for 12 tokens, and a five-token vocabulary. Its output projection is separate from its input embedding table.

Training learns the repeating sequence `red → green → blue`. Training prefix lengths are 4, 7, and 10; validation prefix lengths are 5 and 8, across all three cycle phases. The reserved `<eos>` token is never trained. This is a finite synthetic pattern task, not a natural-language model.

The included checkpoint records training loss `0.0005509947040016146` and validation loss `0.000549696741021241` after 360 Adam updates at learning rate 0.01. These values describe only the supplied synthetic task.

To reproduce and export both the bundled checkpoints and the standalone trained checkpoint:

```sh
node --experimental-strip-types scripts/train-decoder.ts
```

The script writes `src/learning/decoder-checkpoints.json` and `public/checkpoints/mini-decoder.json`. Reproduction is also exercised in the automated suite, so running the export script is unnecessary for normal use.

## Remaining limits

- The new lessons share one tensor engine, but the free builder retains its existing engine. A network can be exported to a numerically equivalent scalar graph; this is a copied prepared state, not a live two-way connection to the lesson. Transformer topology is not exportable to the free builder.
- Free editing is broadest in the graph builder. The network playground supports bounded hidden-layer changes. The decoder's two-block architecture is fixed; its inputs, checkpoint, learned parameter coordinates, and sampling controls are editable. A general transformer architecture editor is not implemented.
- Transformer execution uses recorded forward snapshots with module-aware forward stepping and a top-level phase indicator. It does not present a complete interactive backward/update trace. Training and checkpoint export are available through the reproducible local script; browser transformer training was optional and is not included.
- Lesson workspaces use zoom controls, native touch scrolling, and mouse-drag panning on empty workspace/diagram surfaces. Ordinary node dragging and graph pan/zoom are retained in the free builder; the transformer lesson does not have freely positioned nodes or automatic zoom-triggered expansion. Saved lesson views retain zoom, module expansion, and selections, but do not serialize native scroll offsets.
- Sharing uses downloaded JSON files. There is no hosted sharing service or account system. Navigating to a different preset starts that preset's default state; save a prepared experiment before leaving it.
- The maximum decoder context is 12 tokens. Generation stops at that limit. Checkpoint performance does not establish useful behavior on arbitrary prompts or ordinary language.
- Network training is bounded and runs on the browser's main thread. It is not backed by a worker or GPU; larger training workloads are outside this implementation.
- Responsive laptop/tablet layouts were checked in Chromium. Numeric alternatives to color, keyboard-operable controls, and reduced-motion styles are included. A physical touch-device trial, formal screen-reader/accessibility audit, and classroom projector hardware test were not performed.
- Broad compatibility/migration guarantees for older saved projects are not part of this delivery, following the user's clarification. Existing builder save/import behavior and its loss convention remain covered by the regression tests.
