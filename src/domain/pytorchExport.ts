import { parseArithmetic } from './arithmetic'
import { datasetExamplesForNode, datasetForNode, datasetMode, datasetTargetSlotForNode } from './datasets'
import { inputArityForNode, isLossNode, lossKindForNode, topologicalSort, validateGraph } from './engine'
import { toTensor } from './tensor'
import type { GraphEdge, GraphModel, GraphNode, TensorValue } from './types'

export interface PyTorchExport {
  script: string
  notebook: string
}

const comment = (text: string) => text.replace(/[\r\n#]/g, ' ').trim()
const json = (value: unknown) => JSON.stringify(value)
const identifier = (label: string) => label.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 28) || 'block'

function tensorCode(value: TensorValue | number | undefined): string {
  const tensor = toTensor(value)
  if (!tensor.data.every(Number.isFinite)) throw new Error('Export needs finite parameter and input values.')
  return `torch.tensor(${json(tensor.data)}, dtype=torch.float64).reshape(${json(tensor.shape)})`
}

function sortedInputs(graph: GraphModel, node: GraphNode): GraphEdge[] {
  return graph.edges.filter(edge => edge.target === node.id)
    .sort((a, b) => (a.inputSlot ?? 0) - (b.inputSlot ?? 0) || a.id.localeCompare(b.id))
}

function arithmeticCode(source: string, inputs: string[]): string {
  const parsed = parseArithmetic(source)
  type Expr = typeof parsed.expression
  const render = (expr: Expr): string => {
    switch (expr.kind) {
      case 'number': return json(expr.value)
      case 'input': return inputs[expr.index]
      case 'negate': return `(-${render(expr.child)})`
      case 'binary': return `(${render(expr.left)} ${expr.op === '^' ? '**' : expr.op} ${render(expr.right)})`
    }
  }
  return render(parsed.expression)
}

function operationCode(node: GraphNode, args: string[], graph: GraphModel): string {
  const [a, b, c] = args
  const axis = node.params.axis ?? (node.type === 'concat' ? 1 : 0)
  const kind = node.type === 'tensor-transform' ? node.params.transform ?? 'reshape' : node.type
  switch (kind) {
    case 'matmul': return `torch.matmul(${a}, ${b})`
    case 'multiply': return args.join(' * ')
    case 'add': return args.join(' + ')
    case 'arithmetic': return arithmeticCode(node.params.expression ?? 'x1 * x2', args)
    case 'activation': {
      switch (node.params.activation ?? 'identity') {
        case 'identity': return a
        case 'relu': return `F.relu(${a})`
        case 'sigmoid': return `torch.sigmoid(${a})`
        case 'tanh': return `torch.tanh(${a})`
      }
      break
    }
    case 'embedding': return `F.embedding(${b}.long(), ${a})`
    case 'transpose': return node.params.axes
      ? `${a}.permute(${node.params.axes.join(', ')})`
      : `${a}.permute(*reversed(range(${a}.ndim)))`
    case 'slice': {
      const start = node.params.start ?? 0
      const length = node.params.end === undefined ? `${a}.size(${axis}) - ${start}` : String(node.params.end - start)
      return `${a}.narrow(${axis}, ${start}, ${length})`
    }
    case 'concat': return `torch.cat([${args.map(value => axis === 1 ? `(${value}.reshape(-1, 1) if ${value}.ndim == 1 else ${value})` : value).join(', ')}], dim=${axis})`
    case 'softmax': return `torch.softmax(${a}, dim=-1)`
    case 'causal-mask': return `${a}.masked_fill(torch.triu(torch.ones_like(${a}, dtype=torch.bool), diagonal=1), float('-inf'))`
    case 'layer-norm': return `F.layer_norm(${a}, (${a}.shape[-1],), weight=${b}, bias=${c}, eps=${node.params.epsilon ?? 1e-5})`
    case 'reshape': return `${a}.reshape(${node.params.shape ? json(node.params.shape) : `${a}.shape`})`
    case 'mean': return node.params.axis === undefined
      ? node.params.keepDims ? `${a}.mean().reshape([1] * ${a}.ndim)` : `${a}.mean()`
      : `${a}.mean(dim=${axis}, keepdim=${node.params.keepDims ? 'True' : 'False'})`
    case 'conv2d': return `F.conv2d(${a}.permute(2, 0, 1).unsqueeze(0), ${b}.permute(0, 3, 1, 2), bias=${c}).squeeze(0).permute(1, 2, 0)`
    case 'avgpool2d': return `F.avg_pool2d(${a}.permute(2, 0, 1).unsqueeze(0), 2, stride=2).squeeze(0).permute(1, 2, 0)`
    case 'cross-entropy': return `F.cross_entropy(${a}, ${b}.long().reshape(-1))`
    case 'loss': {
      const loss = lossKindForNode(node, graph)
      switch (loss) {
        case 'squared-error': return `0.5 * torch.sum((${a} - ${b}) ** 2)`
        case 'mse': return `torch.mean((${a} - ${b}) ** 2)`
        case 'mae': return `torch.mean(torch.abs(${a} - ${b}))`
        case 'binary-cross-entropy': return `binary_cross_entropy(${a}, ${b})`
        case 'cross-entropy': return `F.cross_entropy(${a}.reshape(1, -1) if ${a}.ndim == 1 else ${a}, ${b}.long().reshape(-1))`
      }
    }
  }
  throw new Error(`PyTorch export does not support ${node.label} (${node.type}).`)
}

function notebookCell(source: string, kind: 'code' | 'markdown', id: string) {
  return kind === 'code'
    ? { cell_type: 'code', id, metadata: {}, source: source.split('\n').map((line, i, lines) => i < lines.length - 1 ? `${line}\n` : line), execution_count: null, outputs: [] }
    : { cell_type: 'markdown', id, metadata: {}, source: [source] }
}

/** Compile the executable graph, including current weights and dataset, to
 * a self-contained PyTorch program. Visual groups do not change computation. */
export function generatePyTorchExport(graph: GraphModel): PyTorchExport {
  if (graph.nodes.filter(node => node.type === 'dataset').length !== 1) {
    throw new Error('PyTorch export needs exactly one Dataset block.')
  }
  const losses = graph.nodes.filter(isLossNode)
  if (losses.length > 1) throw new Error('PyTorch export supports one Loss block at a time.')
  const issues = validateGraph(graph).filter(issue => issue.code !== 'disconnected')
  if (issues.length) throw new Error(`Finish the graph before exporting: ${issues[0].message}`)
  const order = topologicalSort(graph)
  if (order.length !== graph.nodes.length) throw new Error('Resolve graph cycles before exporting.')
  const datasetNode = graph.nodes.find(node => node.type === 'dataset')!
  const dataset = datasetForNode(datasetNode)
  const examples = datasetExamplesForNode(datasetNode)
  if (!examples.length) throw new Error('The Dataset block has no examples to export.')
  const targetSlot = datasetTargetSlotForNode(datasetNode)
  const nodeById = new Map(graph.nodes.map(node => [node.id, node]))
  const names = new Map(order.map((id, index) => [id, `v_${identifier(nodeById.get(id)!.label)}_${index + 1}`]))
  const sourceName = (edge: GraphEdge): string => {
    const source = nodeById.get(edge.source)!
    const name = names.get(source.id)!
    return source.type === 'dataset' ? `${name}_s${edge.sourceSlot ?? 0}` : name
  }
  const parameterNodes = graph.nodes.filter(node => node.type === 'weight' || node.type === 'bias')
  const parameters = new Map(parameterNodes.map((node, index) => [node.id, `p_${index + 1}`]))
  const parameterLines = parameterNodes.map(node => `        self.${parameters.get(node.id)} = nn.Parameter(${tensorCode(node.params.value)})  # ${comment(node.label)}`)
  const forwardLines: string[] = []
  for (const id of order) {
    const node = nodeById.get(id)!
    const name = names.get(id)!
    const inputs = sortedInputs(graph, node)
    const args = inputs.map(sourceName)
    const expected = inputArityForNode(node)
    if ((node.type !== 'input' && node.type !== 'target') && inputs.length !== expected) {
      throw new Error(`${node.label} has incomplete inputs.`)
    }
    forwardLines.push(`        # ${comment(node.label)} (${node.type})`)
    if (node.type === 'dataset') {
      const count = dataset.featureValues.length + 1
      for (let slot = 0; slot < count; slot++) {
        const feature = slot < targetSlot ? slot : slot - 1
        forwardLines.push(`        ${name}_s${slot} = ${slot === targetSlot ? 'target' : `features[${feature}]`}`)
      }
    } else if (node.type === 'weight' || node.type === 'bias') {
      forwardLines.push(`        ${name} = self.${parameters.get(node.id)}`)
    } else if (node.type === 'input' || node.type === 'target') {
      forwardLines.push(`        ${name} = ${args[0] ?? tensorCode(node.params.value)}`)
    } else {
      forwardLines.push(`        ${name} = ${operationCode(node, args, graph)}`)
    }
  }
  const lossNode = losses[0]
  const predictionEdge = lossNode && sortedInputs(graph, lossNode).find(edge => (edge.inputSlot ?? 0) === 0)
  const actualEdge = lossNode && sortedInputs(graph, lossNode).find(edge => (edge.inputSlot ?? 0) === 1)
  const outputNode = lossNode ? undefined : [...order].reverse().map(id => nodeById.get(id)!).find(node =>
    node.type !== 'dataset' && node.type !== 'target' && !graph.edges.some(edge => edge.source === node.id))
  if (!predictionEdge && !outputNode) throw new Error('Connect an output calculation before exporting.')
  const source = `# Generated by Backprop Builder. Edit TRAIN_EPOCHS and REPORT_EVERY below.\n` +
`import json\nimport random\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\n` +
`torch.set_default_dtype(torch.float64)\n\n` +
`def binary_cross_entropy(prediction, target):\n` +
`    # Match the builder's probability clipping while retaining its derivative.\n` +
`    clipped = prediction.clamp(1e-7, 1 - 1e-7)\n` +
`    probability = prediction + (clipped - prediction).detach()\n` +
`    return -(target * torch.log(probability) + (1 - target) * torch.log(1 - probability)).mean()\n\n` +
`class BuilderModel(nn.Module):\n` +
`    def __init__(self):\n` +
`        super().__init__()\n` +
`${parameterLines.length ? parameterLines.join('\n') : '        pass'}\n\n` +
`    def forward(self, features, target):\n` +
`${forwardLines.join('\n')}\n` +
`        return ${predictionEdge ? sourceName(predictionEdge) : names.get(outputNode!.id)}, ${lossNode ? names.get(lossNode.id) : 'None'}, ${actualEdge ? sourceName(actualEdge) : 'target'}\n\n` +
`DATASET = json.loads(${json(json({
  label: dataset.label,
  task: dataset.task,
  classLabels: dataset.classLabels,
  vocabulary: dataset.vocabulary,
  examples: examples.map(example => ({ label: example.label, split: example.split, features: example.features, target: example.target })),
}))})\n` +
`BATCH_MODE = ${datasetMode(datasetNode) === 'batch' ? 'True' : 'False'}\n` +
`LEARNING_RATE = ${json(graph.learningRate)}\n` +
`TRAIN_EPOCHS = ${lossNode ? 10 : 0}  # Add a Loss block to enable training.\nREPORT_EVERY = 1\n\n` +
`def tensor_value(value):\n` +
`    return torch.tensor(value['data'], dtype=torch.float64).reshape(value['shape'])\n\n` +
`def model_inputs(rows):\n` +
`    if BATCH_MODE:\n` +
`        features = [torch.tensor([row['features'][i]['data'][0] for row in rows]) for i in range(len(rows[0]['features']))]\n` +
`        target = torch.tensor([row['target']['data'][0] for row in rows])\n` +
`        return features, target\n` +
`    row = rows[0]\n` +
`    return [tensor_value(value) for value in row['features']], tensor_value(row['target'])\n\n` +
`def batches(rows):\n` +
`    return [rows] if BATCH_MODE else [[row] for row in rows]\n\n` +
`def evaluate(model, rows):\n` +
`    if not rows:\n` +
`        return None, None, []\n` +
`    model.eval()\n` +
`    losses, pairs, hits = [], [], []\n` +
`    with torch.no_grad():\n` +
`        for batch in batches(rows):\n` +
`            features, target = model_inputs(batch)\n` +
`            prediction, loss, actual_target = model(features, target)\n` +
`            if loss is not None:\n` +
`                losses.append(float(loss))\n` +
`            actual = actual_target.reshape(-1)\n` +
`            if prediction.numel() % actual.numel() != 0:\n` +
`                continue\n` +
`            scores = prediction.reshape(actual.numel(), -1)\n` +
`            categorical = DATASET['task'] in ('classification', 'binary-classification', 'sequence')\n` +
`            if categorical and scores.shape[-1] > 1:\n` +
`                predicted = scores.argmax(dim=-1)\n` +
`            elif DATASET['task'] == 'binary-classification':\n` +
`                predicted = (scores[:, 0] >= 0.5).long()\n` +
`            else:\n` +
`                predicted = scores[:, 0]\n` +
`            for index, value in enumerate(predicted):\n` +
`                label = batch[index]['label'] if BATCH_MODE else batch[0]['label']\n` +
`                pairs.append((label, actual[index].item(), value.item()))\n` +
`                if categorical:\n` +
`                    hits.append(int(value.item() == actual[index].item()))\n` +
`    return (sum(losses) / len(losses) if losses else None), (sum(hits) / len(hits) if hits else None), pairs\n\n` +
`model = BuilderModel()\n` +
`train_rows = [row for row in DATASET['examples'] if row['split'] == 'train']\n` +
`test_rows = [row for row in DATASET['examples'] if row['split'] == 'test']\n` +
`if not train_rows:\n` +
`    raise ValueError('The dataset has no training examples.')\n` +
`parameters = list(model.parameters())\n` +
`optimizer = torch.optim.SGD(parameters, lr=LEARNING_RATE) if parameters else None\n` +
`for epoch in range(1, TRAIN_EPOCHS + 1):\n` +
`    model.train()\n` +
`    order = list(train_rows)\n` +
`    if not BATCH_MODE:\n` +
`        random.Random(42 + epoch).shuffle(order)\n` +
`    for batch in batches(order):\n` +
`        features, target = model_inputs(batch)\n` +
`        if optimizer is not None:\n` +
`            optimizer.zero_grad()\n` +
`        _, loss, _ = model(features, target)\n` +
`        if optimizer is not None:\n` +
`            loss.backward()\n` +
`            optimizer.step()\n` +
`    if epoch % REPORT_EVERY == 0 or epoch == TRAIN_EPOCHS:\n` +
`        train_loss, train_accuracy, _ = evaluate(model, train_rows)\n` +
`        print(f'Epoch {epoch}: train loss={train_loss:.6g}' + (f', accuracy={train_accuracy:.1%}' if train_accuracy is not None else ''))\n` +
`test_loss, test_accuracy, predictions = evaluate(model, test_rows)\n` +
`if test_loss is not None:\n` +
`    print(f'Test loss={test_loss:.6g}' + (f', accuracy={test_accuracy:.1%}' if test_accuracy is not None else ''))\n` +
`if predictions:\n` +
`    print('Test predictions (first 10):')\n` +
`    for label, actual, predicted in predictions[:10]:\n` +
`        print(f'  {label}: predicted={predicted:g}, actual={actual:g}')\n`

  const notebook = json({
    cells: [
      notebookCell(`# ${dataset.label} — Backprop Builder export\n\nRun the cells to ${lossNode ? 'train and test' : 'evaluate'} this graph in PyTorch. Current parameter values and dataset examples are included.`, 'markdown', 'intro'),
      notebookCell(source, 'code', 'model-and-training'),
    ],
    metadata: { kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' }, language_info: { name: 'python' } },
    nbformat: 4,
    nbformat_minor: 5,
  })
  return { script: source, notebook }
}
