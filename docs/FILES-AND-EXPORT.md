# Saving and exporting

[Guide home](../README.md) · [Get started](USAGE.md) · [Example models](EXAMPLES.md)

## Keep an editable Backprop Builder project

Choose **File → Save** to download a project JSON file. It includes the graph, node settings and parameters, group hierarchy, current view and selection, and the recorded execution state. Choose **File → Import** to reopen it. Import also opens any [example project](../public/models/README.md). **File → New** clears the workspace; save the current project first if you want to keep it. Projects are local files rather than accounts or hosted documents.

Use **Edit → Undo**, Copy, Paste, and Duplicate while building. The app does not promise compatibility with every historical saved-project format.

## Export runnable PyTorch

**File → Export Python file** downloads `backprop-builder-model.py`. **File → Export PyTorch notebook** downloads `backprop-builder-model.ipynb`. The export turns a supported connected graph into PyTorch tensor calculations with its current parameter values and dataset. A graph must have exactly one Dataset block, complete connections, and at most one Loss block. If it cannot be translated, the app shows an export error instead of a misleading file.

The generated program is a starting point for experiments outside the visual app. It includes a small training loop when the graph has a loss and emits evaluation results. It is not a hosted training service or an export of the canvas layout.

## Open in Google Colab

Choose **File → Open in Colab**. The app downloads the notebook and opens a Colab tab. In Colab, choose **File → Upload notebook**, then select the downloaded `backprop-builder-model.ipynb`. Colab may ask you to connect a runtime before running cells. The browser cannot silently upload a local notebook to your Google account, so the upload step is intentional.

If you only want the guide, choose **Backprop Builder → Reference** in the app's top bar. It opens the GitHub README in a new tab.
