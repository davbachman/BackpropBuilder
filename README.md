# Backprop Builder

Backprop Builder is a browser-based teaching app for small tensor backpropagation. Students build a tiny feed-forward computation graph, step through the forward pass, inspect squared-error loss, watch gradients flow backward, and apply gradient descent updates to trainable weights and biases. Scalar values are represented as tensors with shape `[]`, so the starter graph stays compact while using the tensor engine.

## Run Locally

```bash
npm install
npm run dev
```

Open the local Vite URL printed by the command.

## Useful Commands

```bash
npm test
npm run lint
npm run build
```

## What Is Included

- React + TypeScript + Vite app with all computation in the browser.
- React Flow graph canvas with custom operation nodes and labeled edges.
- Starter graph: `x -> x * w -> + b -> sigmoid -> squared error loss`.
- Node types: input, weight, bias, multiply, add, activation, target, and loss.
- Source nodes accept scalar values or JSON-style tensor literals such as `[1, 2, 3]`.
- Add, multiply, and activation nodes apply elementwise tensor operations with scalar broadcasting.
- Squared-error loss reduces tensor prediction errors to one scalar objective for gradient descent.
- Step controls for forward pass, loss, backward pass, and parameter updates.
- Math, gradient, and pseudocode overlay toggles.
- JSON project state save/import for continuing a workspace later.

## Project Structure

```text
src/domain/
  engine.ts       Tensor autodiff, validation, topological sort, updates
  examples.ts     Starter graph factories
  session.ts      JSON project state creation, download, and import parsing
  tensor.ts       Tensor representation, formatting, parsing, broadcasting
  types.ts        Shared graph and project-state types

src/components/
  GraphCanvas.tsx React Flow workspace
  BuilderNode.tsx Custom graph node UI
  BuilderEdge.tsx Custom edge labels and phase styling
```

The numerical engine is intentionally small. Elementwise tensors and scalar broadcasting are supported; matrix multiplication, datasets, convolutional layers, and PyTorch export remain out of scope.
