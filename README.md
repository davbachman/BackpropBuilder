# Backprop Builder

[Open the app](https://davbachman.github.io/BackpropBuilder/)

Created by David Bachman with GPT 5.5. To learn more about David see https://pzacad.pitzer.edu/~dbachman/, and subscribe to his AI podcast *Entropy Bonus* at https://profbachman.substack.com/.

## Brief description
BackpropBuilder is a local visual teaching studio: open a neuron, trace a network, inspect attention, and generate tokens with a real miniature transformer. Ten presets connect individual arithmetic to networks and a two-block decoder. The editable computation-graph builder remains available with persistent nested modules and backpropagation.

## Instructions for use
Run `npm install` and `npm run dev`, then open the displayed local URL. Choose a preset, predict an outcome, run the calculation, and open a component to follow its numbers. Use **Blank builder** for free graph editing. Save prepared experiments as JSON to share or restore them.

See the [usage guide](docs/USAGE.md) for lesson workflows and the [validation report](docs/VALIDATION.md) for verified behavior and remaining limits. The decoder learns only a synthetic repeating color sequence; it does not understand ordinary language.

## Development

```sh
npm test
npm run build
npm run lint
npm run train:decoder
```

Checkpoint training requires Node.js with native TypeScript stripping (tested on Node 25). The deterministic script exports the bundled initial/trained checkpoints and `public/checkpoints/mini-decoder.json`. Normal app use needs no account, server inference, or large model download.
