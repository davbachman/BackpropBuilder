# Neural Canvas audit fixes — September 28, 2026

All 13 confirmed defects in [the original audit](AUDIT-2026-09-28.md) have been addressed. The implementation also improves evaluation, exports, canvas rendering, dependency hygiene, and deployment checks. These changes are local; no commit, push, or deployment was performed.

## Correctness repairs

| Audit finding | Implemented repair |
| --- | --- |
| 1. Old training can overwrite a replacement workspace | Training and inference own an abort controller; every asynchronous publication checks ownership. Model edits, import, New, and undo cancel obsolete work. Results merge into the current layout so moving or renaming nodes during a run is preserved. |
| 2. Tensor batch axes alter broadcasting and gradients | Arithmetic and loss broadcasting align per-example dimensions separately from the batch axis. Shared constant predictions are expanded across examples. Added trace, tensor, and PyTorch parity checks across batch sizes. |
| 3. Binary two-logit models fail or misdecode | Tensor cross entropy aligns class-index targets correctly; inference uses argmax for categorical logits and thresholds scalar binary probabilities. |
| 4. Projected inline edits do not update the model | Projected scalar weight/bias edits resolve to canonical tensor coordinates and support undo. Unsupported synthetic controls are read-only. |
| 5. Canvas callbacks restore obsolete settings | Canvas callbacks refresh with the current graph while preserving live drag positions. Large JSON serialization guards were removed. |
| 6. Inference reports survive computational edits | Shared invalidation clears reports and recorded execution after parameter, activation, loss, dataset, and structural changes. |
| 7. Export cannot read the original CSV split column | Generated Python identifies and removes the original split column, including duplicate-header cases. End-to-end tests execute against original input files. |
| 8. Exported metrics include regularization penalties | Evaluation reports data loss; optimization retains the regularized objective. |
| 9. Disconnected parameters crash exported training | Generated training checks whether the loss is differentiable before backpropagating. Empty or unused parameter sets are handled. |
| 10. Sequence accuracy exceeds 100% | Tensor evaluation counts target positions, including built-in sequence examples, and respects sequence masks. |
| 11. Modifier-click loses selection | Canvas click handling preserves React Flow's accumulated modifier selection. |
| 12. Code navigation fails on ungrouped graphs | Focus falls back to node bounds when semantic group geometry is absent. |
| 13. Hidden module contents remain keyboard-accessible | Covered contents are inert and hidden from assistive technology until accessible. |

Additional deferred-import regressions cover Cancel, Escape, unmount, replacement reads, competing project/text imports, and imports superseded by training. Old completion, error, and cleanup handlers cannot overwrite newer work or clear its busy state.

## Performance changes

Measurements below are local, targeted benchmarks, not guarantees for every model or browser.

| Path | Measured change |
| --- | --- |
| Forward evaluation, 10,000 examples / 26-node graph | 853.5 ms to 68.9 ms, approximately 12.4× faster, with identical loss. Topology is compiled once; metrics skip gradients and trace caches. |
| Interactive dataset reporting | Evaluation yields between chunks and accepts cancellation; loss-only reports avoid constructing prediction rows. A single expensive example remains synchronous. |
| Count-vector export, 1,000 documents / 4,000 vocabulary entries | 8.284 MB to 0.305 MB, approximately 96% smaller. Python reconstructs sparse vectors per example. |
| CSV classification, 50,000 unique labels | 2,084 ms to 37.5 ms by mapping labels once instead of repeatedly scanning them. |
| 256×256 transpose | 9.8–17.5 ms to 1.8–3.4 ms by precomputing strides. |
| Decoder overview tooltip text | 45,943 to 10,597 characters, approximately 77% less, by deferring inaccessible content and memoizing formatting. |
| Initial JavaScript with the same patched dependencies | 242,856 to 232,827 bytes gzip, approximately 4% smaller, by deferring optional CNN, decoder, and text-generation controls and checkpoint imports. Final builds vary slightly with subsequent code changes. |

## Verification and delivery checks

- **732 tests passed across 65 files; zero failures and zero skipped tests**, with the available PyTorch runtime enabled. This includes generated Python execution and native tensor/PyTorch numerical parity.
- **TypeScript, production build, and ESLint passed.**
- **Full `npm audit`: zero reported vulnerabilities**, including development dependencies. Vite and Vitest were updated to compatible patched releases and the lockfile regenerated; `npm ci` succeeded.
- Browser verification covered trace training, inference, clearing stale reports after an edit, and discarding an active run after File → New. The production build also completed two Tensor CPU epochs with batch size five and MSE loss, followed by held-out inference.
- Deployment now requires lint, tests, build, and a separate pinned Python/PyTorch parity job. These gates have been added locally; GitHub Actions has not been run remotely.
- The existing code-navigation test now waits for its camera transition instead of assuming synchronous selection.

Verification artifacts are in the ignored `output/audit/` directory, including `fix-verification.json`, `fix-build.log`, and `fix-dependencies.json`.

## Remaining limits

The build still warns about large main and TensorFlow chunks. Further reductions would require broader loading or backend changes. WebGL/WebGPU execution and exhaustive mobile/cross-browser behavior were not independently verified. The existing tensor engine supports MSE, MAE, binary cross entropy, and cross entropy; the Starter's default squared-error loss continues to use the trace engine. Tensor-training settings remain outside the existing Python-export feature set.

During browser checks, a long-lived development page requested an expired optimized dependency URL and received HTTP 504. Production lazy loading succeeded. Git CLI inspection remains unavailable because the system Xcode license has not been accepted; no license or system settings were changed.
