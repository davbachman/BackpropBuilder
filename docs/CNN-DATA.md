# Handwritten digits and the miniature CNN

The image-classification preset uses 8 × 8 images from **Optical Recognition of Handwritten Digits**, created by E. Alpaydin and C. Kaynak (1998), UCI Machine Learning Repository, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49). UCI distributes this dataset under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

Source: [UCI dataset and attribution](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). The source file is the [scikit-learn public copy](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/data/digits.csv.gz), documented by [`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). This is a copy of UCI's original test partition. It is a different dataset from MNIST's 28 × 28 images.

Changes for this app: the deterministic preparation script shuffles each class with a seeded generator, takes 50 images per class, and divides them into 40 training and 10 held-out images. The 500 records retain their original row identifiers and integer pixel intensities from 0 to 16. This is a new educational split, not the original writer-independent UCI split. The graph divides each pixel by 16 before computing. The app includes 400 training images and 100 held-out images with no overlap.

## Reproduction

The committed `src/learning/digits.json` is sufficient for offline training. To refresh the subset from the public mirror, run `node scripts/prepare-digits.mjs`. The split seed is 37. To reproduce the included checkpoint, run `node --experimental-strip-types scripts/train-cnn.ts`.

The model uses a valid 3 × 3 convolution with four filters and one input channel, ReLU, 2 × 2 average pooling with stride two, a flattened vector of 36 features, and a linear ten-class head with softmax. There are **410 trainable parameters**: 36 convolution weights, four convolution biases, 360 classifier weights, and ten classifier biases. The graph uses cross-entropy on logits for stable backpropagation.

The checkpoint uses seed 19, 180 fixed epochs, mini-batches of 20, and Adam with learning rate 0.006. No pretrained external weights are used. The committed checkpoint classifies all 400 training images correctly and **97 of the 100 held-out images** correctly. These counts describe this included small split and checkpoint, not general recognition performance. Editing a filter or training further changes the current model; the panel labels the historical checkpoint result separately.

The filter visualizer reads the same canonical tensors used by the canvas. Editing a kernel cell changes one parameter shared across all spatial locations. Selecting a feature-map cell highlights its actual 3 × 3 receptive field in the input. The small pooled maps and prediction bars come from the graph's computed values. Gradients accumulate from every use of each shared filter weight.
