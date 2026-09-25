# Saving and exporting

[Guide home](../README.md) · [Get started](USAGE.md) · [Example models](EXAMPLES.md)

## Keep an editable Backprop Builder project

Choose **File → Save** to download a project JSON file. It includes the graph, node settings and parameters, group hierarchy, current view and selection, and the recorded execution state. Choose **File → Import** to reopen it. Import also opens any [example project](../public/models/README.md). **File → New** clears the workspace; save the current project first if you want to keep it. Projects are local files rather than accounts or hosted documents.

Use **Edit → Undo**, Copy, Paste, and Duplicate while building. The app does not promise compatibility with every historical saved-project format.

## Export runnable PyTorch

**File → Export Python file** downloads `backprop-builder-model.py`. **File → Export PyTorch notebook** downloads `backprop-builder-model.ipynb`. The export turns a supported connected graph into PyTorch tensor calculations with its current parameter values and dataset. A graph must have exactly one Dataset block, complete connections, and at most one Loss block. If it cannot be translated, the app shows an export error instead of a misleading file.

The generated program is a starting point for experiments outside the visual app. It includes a small training loop when the graph has a loss and emits evaluation results. The current **Train** settings carry over: batch size, reshuffling, epoch count, and reporting interval. At each report it evaluates training and held-out loss without updating parameters on the held-out examples, and plots both curves when Matplotlib is available. It is not a hosted training service or an export of the canvas layout.

## Open in Google Colab

Without a configured connection, **File → Open in Colab** downloads the notebook and opens Colab. In Colab, choose **File → Upload notebook**, then select `backprop-builder-model.ipynb`.

## Direct Colab connection

With a Google Drive connection, **File → Open in Colab** uploads the generated notebook to your Drive and opens that file in a new Colab tab. This is a direct browser-to-Drive upload; the app does not need a server or receive your Google password. Each export creates a new notebook file in your Drive. The notebook may include your model parameters and dataset, so choose this action only when you want those contents in your Google account.

To configure it for one browser:

1. In Google Cloud, create or select a project, enable the **Google Drive API**, configure the OAuth consent screen, and create an OAuth **Web application** client. Add `https://davbachman.github.io` as an authorized JavaScript origin. For local development, also add your Vite origin, such as `http://127.0.0.1:5173`.
2. In the app, open **File → Colab connection…** and enter the **client ID** ending in `.apps.googleusercontent.com`. Do not enter a client secret. The ID is saved in this browser.
3. Choose **Connect Google Drive** and approve Google's `drive.file` permission. This lets the app create and manage files it creates, without broad access to the rest of your Drive.
4. Choose **Upload & open current notebook** there, or later use **File → Open in Colab**. If browser popup blocking prevents the tab from opening, the app provides an **Open notebook** link after upload.

Access tokens remain in memory only; reconnect after a reload or token expiry. **Disconnect** revokes the current token. When no connection is configured, the download-and-upload path above remains available. Google may restrict access to OAuth test users until the Cloud project's consent screen is published.

For a shared class deployment, the repository owner can set the GitHub Actions variable `GOOGLE_OAUTH_CLIENT_ID` to the same public web client ID. The Pages build then supplies it as the default for every browser; students still approve the Drive permission themselves. A locally saved client ID overrides that default.

See Google's [web authorization setup](https://developers.google.com/identity/oauth2/web/guides/use-token-model), [Drive file scope](https://developers.google.com/workspace/drive/api/guides/api-specific-auth), and [upload API](https://developers.google.com/workspace/drive/api/guides/manage-uploads) for the current platform requirements. Colab may ask you to connect a runtime before running cells.

If you only want the guide, choose **Backprop Builder → Reference** in the app's top bar. It opens the GitHub README in a new tab.
