# Development

[Guide home](../README.md) · [Example models](EXAMPLES.md) · [CNN data attribution](CNN-DATA.md)

Backprop Builder is a React and TypeScript app built with Vite. The graph runs in the browser; normal use does not require an account, a model-serving API, or a large model download.

From the repository root:

```sh
npm install
npm run dev
```

Open the local URL printed by Vite. A production build is created with `npm run build`. The repository checks are `npm test` and `npm run lint`. After editing model definitions, `npm run export:models` regenerates the importable project JSON files.

See [Validation and scope](VALIDATION.md) for what those checks cover and the model-size limits.

Checkpoint generation is separate from normal browser use. `npm run train:decoder` writes the miniature decoder checkpoint; `node --experimental-strip-types scripts/train-cnn.ts` reproduces the included CNN checkpoint. These scripts need Node.js with native TypeScript stripping (the project has been tested with Node 25).

The image model uses a subset of the [UCI Optical Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), distributed through scikit-learn. The detailed source and reproduction notes are in [CNN-DATA.md](CNN-DATA.md). It is an 8×8 digit task, not MNIST. The transformer uses a small synthetic color-cycle sequence, not a natural-language corpus.
