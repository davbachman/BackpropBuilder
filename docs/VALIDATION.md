# Validation and scope

[Guide home](../README.md) · [Development](DEVELOPMENT.md) · [Example models](EXAMPLES.md)

Run the repository checks from the project root:

```sh
npm test
npm run build
npm run lint
```

The automated suite covers graph execution and gradients, tensor shapes and broadcasting, all public importable projects, dataset-backed training and evaluation, project save/import, PyTorch export, and key canvas and sidebar interactions. A build verifies TypeScript and the static Vite bundle; lint checks the source. These checks do not substitute for inspecting a complex model in a real browser.

The app is intended for small, explainable models. The decoder learns a synthetic color sequence with a 12-token context and does not understand natural language. The CNN uses normalized UCI 8×8 digits rather than MNIST. Running large graphs or datasets entirely in a browser can be slow; PyTorch export is available for supported connected graphs when more experimentation is needed.

The importable examples and your own projects use the same editable graph. Groups are views of contained calculations. A save creates local JSON, and importing another file replaces the current workspace, so save work you want to retain. Saved-project compatibility across older formats is not guaranteed.

For the image data's attribution and checkpoint reproduction, see [CNN-DATA.md](CNN-DATA.md).
