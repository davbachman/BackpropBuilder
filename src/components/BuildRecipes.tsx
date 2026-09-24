export function BuildRecipes() {
  return <div className="build-recipes">
    <details><summary>Build a CNN from scratch</summary><ol>
      <li>Add <b>Dataset</b>, choose Handwritten digits. Its outputs are an <code>8×8×1</code> image and a class ID.</li>
      <li>Add Weight <code>4×3×3×1</code>, initialize with He; add Bias <code>4</code>, initialize to zeros. Connect image, filters and bias to Convolution ports 1, 2 and 3.</li>
      <li>Connect Activation (ReLU) → Average pooling → Reshape <code>1, 36</code>.</li>
      <li>Matrix product with Xavier Weight <code>36×10</code>, then Add Bias <code>10</code>. These are ten raw class scores.</li>
      <li>Connect scores and the dataset’s digit output to Cross-entropy. Optionally branch scores to Softmax. Select the dataset to train and test; select Convolution to inspect filters.</li>
    </ol></details>
    <details><summary>Build a transformer from scratch</summary><ol>
      <li>Choose the Color cycle dataset (5 tokens). Initialize token Weight <code>5×8</code> and position Weight <code>12×8</code> with Xavier. Wire each table and its dataset IDs into an Embedding lookup. Add the two outputs.</li>
      <li>Layer norm takes this stream, Weight <code>8</code> initialized to ones, and Bias <code>8</code> initialized to zeros.</li>
      <li>Build three Matrix products with independent Xavier Weights <code>8×8</code> to obtain Q, K and V. For each of two heads, Slice all three on axis 1: <code>0…4</code> or <code>4…8</code>.</li>
      <li>Per head: transpose K, multiply Q by Kᵀ, then Multiply by an Input constant <code>0.5</code> (= 1/√4). Connect Causal mask → Softmax → Matrix product with V.</li>
      <li>Concatenate head outputs on axis 1; project with an <code>8×8</code> Weight. Add the original stream as a residual connection.</li>
      <li>Layer norm again, then an MLP: Matrix product <code>8×16</code> → Add Bias <code>16</code> → ReLU → Matrix product <code>16×8</code> → Add Bias <code>8</code>. Add another residual.</li>
      <li>Select and group a layer or complete block; name it in the inspector. Copy and reconnect blocks to build a deeper network.</li>
      <li>Final Layer norm → Matrix product with Weight <code>8×5</code> → Add Bias <code>5</code>. Wire these logits and dataset next-token IDs into Cross-entropy. Select the dataset to train, evaluate and generate.</li>
    </ol><p>Start with a learning rate around 0.005. Counting uses 7 vocabulary entries, so its embedding and output projection need 7 instead of 5. Shapes describe one sequence; its length may change.</p></details>
  </div>
}
