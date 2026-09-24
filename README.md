# Backprop Builder

[Open the app](https://davbachman.github.io/BackpropBuilder/)

Created by David Bachman with GPT 5.5. To learn more about David see https://pzacad.pitzer.edu/~dbachman/, and subscribe to his AI podcast *Entropy Bonus* at https://profbachman.substack.com/.

## Brief description
BackpropBuilder is an editable neural-network canvas. Zoom from a complete model into layers, neurons, and individual calculations; watch values flow forward and gradients flow backward. Eleven presets include dataset-backed regression and classification, attention, a working two-block transformer, and an 8×8 handwritten-digit CNN with editable filters and feature maps.

## Instructions for use
Run `npm install` and `npm run dev`, then open the displayed local URL. Choose a preset and zoom into its connected architecture. Click a wire for its values, or a calculation to edit its parameters. Use **Blank builder** to build operations, group a neuron, and copy it into a larger network. **File → Save** downloads the working graph as JSON.

See the [usage guide](docs/USAGE.md) for lesson workflows and the [validation report](docs/VALIDATION.md) for verified behavior and remaining limits. The decoder learns only a synthetic repeating color sequence; it does not understand ordinary language.

## Development

```sh
npm test
npm run build
npm run lint
npm run train:decoder
node --experimental-strip-types scripts/train-cnn.ts
```

Checkpoint training requires Node.js with native TypeScript stripping (tested on Node 25). The deterministic script exports the bundled initial/trained checkpoints and `public/checkpoints/mini-decoder.json`. Normal app use needs no account, server inference, or large model download.

The digit model uses the UCI optical-recognition dataset, not MNIST; see [dataset attribution and CNN reproduction](docs/CNN-DATA.md).
