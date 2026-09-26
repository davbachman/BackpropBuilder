Created by David Bachman with Codex

# Backprop Builder

[Open Backprop Builder](https://davbachman.github.io/BackpropBuilder/)

Backprop Builder is a visual, editable machine-learning workbench for students. Start with a blank canvas, connect individual calculations into a model, and watch values move forward and gradients move backward. Group calculations into neurons, layers, attention heads, or larger modules, then zoom between those scales. You can train and test on built-in or imported data and export a working PyTorch version of a supported graph.

## Guide

| Page | What it covers |
| --- | --- |
| [Get started](docs/USAGE.md) | A first model, the sidebars, and the route from blank canvas to a training run |
| [Canvas and navigation](docs/CANVAS.md) | Adding, connecting, moving, grouping, zooming, and using the code outline |
| [Block reference](docs/BLOCKS.md) | Every block in the palette, its inputs, and common tensor shapes |
| [Build models from scratch](docs/BUILDING-MODELS.md) | Recipes for an MLP, attention, a transformer block, and a digit CNN |
| [Datasets](docs/DATASETS.md) | Built-in data, custom CSVs, feature and target wiring, batches, and train/test splits |
| [Training, testing, and reports](docs/TRAINING.md) | Backpropagation, epoch runs, inference, predictions, accuracy, and troubleshooting |
| [Example models and lessons](docs/EXAMPLES.md) | Importable projects from a neuron through attention, a transformer, and a digit CNN |
| [Saving and exporting](docs/FILES-AND-EXPORT.md) | Project JSON, PyTorch files and notebooks, and dataset files |
| [Development](docs/DEVELOPMENT.md) | Local setup, checks, model generation, and dataset attribution |

The [importable model files](public/models/README.md) are editable projects made with the same blocks as the blank canvas. The full decoder is a small synthetic sequence model, and the image classifier uses UCI 8×8 digits rather than MNIST.

Learn more about [David Bachman](https://pzacad.pitzer.edu/~dbachman/) and his AI podcast, [*Entropy Bonus*](https://profbachman.substack.com/).
