# Importable model files

Backprop Builder opens to a blank canvas. Download any JSON file in this folder, then choose **File → Import** in the app and select it. Use **File → Save** to download your changes; you can import that file later.

The files are complete editable projects, not screenshots or locked demonstrations. They use the same nodes, connections, datasets, and controls available on a blank canvas.

| File | Model |
| --- | --- |
| [starter-neuron.json](starter-neuron.json) | A small dataset-backed neuron |
| [linear.json](linear.json) | Linear regression |
| [neuron.json](neuron.json) | ReLU neuron |
| [small-network.json](small-network.json) | Small neural network |
| [playground.json](playground.json) | Two-dimensional classifier |
| [probabilities.json](probabilities.json) | Scores and softmax |
| [embeddings.json](embeddings.json) | Token and position embeddings |
| [attention.json](attention.json) | One-query attention |
| [causal.json](causal.json) | Causal self-attention |
| [block.json](block.json), [block-untrained.json](block-untrained.json) | Transformer block with trained or initial weights |
| [decoder.json](decoder.json), [decoder-untrained.json](decoder-untrained.json) | Full miniature decoder with trained or initial weights |
| [cnn.json](cnn.json) | Digit CNN |

To regenerate these files after changing a model definition, run `npm run export:models` from the project root.
