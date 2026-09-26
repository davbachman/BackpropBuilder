# Saving and exporting

[Guide home](../README.md) · [Get started](USAGE.md) · [Example models](EXAMPLES.md)

## Keep an editable Backprop Builder project

Choose **File → Save** to download a project JSON file. It includes the graph, node settings and parameters, group hierarchy, current view and selection, and the recorded execution state. Choose **File → Import** to reopen it. Import also opens any [example project](../public/models/README.md). **File → New** clears the workspace; save the current project first if you want to keep it. Projects are local files rather than accounts or hosted documents.

Use **Edit → Undo**, Copy, Paste, and Duplicate while building. The app does not promise compatibility with every historical saved-project format.

## Export runnable PyTorch

**File → Export Python file** downloads `backprop-builder-model.py`. **File → Export PyTorch notebook** downloads `backprop-builder-model.ipynb`. The export turns a supported connected graph into PyTorch tensor calculations with its current parameter values. A graph must have exactly one Dataset block, complete connections, and at most one Loss block. If it cannot be translated, the app shows an export error instead of a misleading file.

Dataset rows are not embedded in the Python file or notebook. For an imported CSV, put the original CSV beside the downloaded file, keeping its filename and row order. For a built-in dataset, the app downloads `backprop-builder-dataset.json` alongside the model export. Keep both downloads in the same folder. The generated code loads the sibling dataset file before creating and running the model. In a notebook, run it with that folder as the working directory.

The generated program is a starting point for experiments outside the visual app. It includes a small training loop when the graph has a loss and emits evaluation results. The current **Train** settings carry over: batch size, reshuffling, epoch count, and reporting interval. At each report it evaluates training and held-out loss without updating parameters on the held-out examples, and plots both curves when Matplotlib is available. It is not a hosted training service or an export of the canvas layout.

If you only want the guide, choose **Backprop Builder → Reference** in the app's top bar. It opens the GitHub README in a new tab.
