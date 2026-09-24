import { getCheckpoint } from '../learning/decoder'
import { createCnnPreset } from '../learning/cnn'
import { createNetwork, networkToGraph } from '../learning/network'
import type { NetworkKind } from '../learning/network'
import type { LessonKind } from '../learning/presets'
import type {
  ActivationKind,
  DatasetKind,
  GraphGroup,
  GraphModel,
  NodeParams,
  NodeType,
  TensorValue,
} from './types'

export type ModelPresetKind = LessonKind | 'blank'
export type ModelCheckpoint = 'trained' | 'untrained'

const value = (shape: number[], data: number[]): TensorValue => ({
  shape,
  data: [...data],
})

/** All levels are representations of this same executable graph. Groups only
 * control its appearance; they never substitute another numerical model. */
class ModelGraph {
  graph: GraphModel = {
    nodes: [],
    edges: [],
    groups: [],
    learningRate: 0.03,
    view: { expandedGroupIds: [], semanticZoom: true },
  }

  node(
    id: string,
    type: NodeType,
    label: string,
    inputs: string[],
    x: number,
    y: number,
    params: NodeParams = {},
  ): string {
    this.graph.nodes.push({ id, type, label, position: { x, y }, params })
    inputs.forEach((source, inputSlot) =>
      this.graph.edges.push({
        id: `${source}→${id}:${inputSlot}`,
        source,
        target: id,
        inputSlot,
      }),
    )
    return id
  }

  source(
    id: string,
    type: 'input' | 'weight' | 'bias' | 'target',
    label: string,
    data: number | TensorValue,
    x: number,
    y: number,
  ): string {
    return this.node(id, type, label, [], x, y, { value: data })
  }

  group(
    id: string,
    label: string,
    kind: string,
    start: number,
    x: number,
    y: number,
    parentId?: string,
    detail?: Record<string, unknown>,
  ): GraphGroup {
    const group: GraphGroup = {
      id,
      label,
      kind,
      nodeIds: this.graph.nodes.slice(start).map((node) => node.id),
      position: { x, y },
      dimensions: { width: 220, height: 160 },
      parentId,
      detail,
    }
    this.graph.groups!.push(group)
    return group
  }
}

export function createModelPreset(
  kind: ModelPresetKind,
  checkpoint: ModelCheckpoint = 'trained',
): GraphModel {
  if (kind === 'cnn') return createCnnPreset()
  if (kind === 'blank') return { ...new ModelGraph().graph, view: { expandedGroupIds: [] } }
  if (
    kind === 'linear' ||
    kind === 'neuron' ||
    kind === 'small-network' ||
    kind === 'playground'
  )
    return attachPresetDataset(scalarNetworkPreset(kind), kind === 'playground' ? 'xor' : kind === 'small-network' ? 'plane-2d' : 'line-1d', kind === 'linear' || kind === 'neuron' ? ['input-0', 'target'] : ['input-0', 'input-1', 'target'], 9)
  if (kind === 'probabilities') return attachPresetDataset(probabilitiesPreset(), 'class-scores', ['scores', 'target'])
  if (kind === 'attention' || kind === 'causal')
    return attachPresetDataset(attentionPreset(kind === 'causal'), kind === 'causal' ? 'attention-sequence' : 'attention-query', ['queries', 'keys', 'values'])
  return attachPresetDataset(decoderPreset(kind, checkpoint), 'color-cycle', ['token-ids', 'position-ids', 'decoder-targets'])
}

function scalarNetworkPreset(kind: NetworkKind): GraphModel {
  const model = createNetwork(kind)
  if (kind === 'neuron') model.layers[0].activation = 'relu'
  const graph = networkToGraph(model, {
    id: 'canvas-example',
    x: kind === 'linear' || kind === 'neuron' ? [1] : [0.6, -0.2],
    y: kind === 'playground' ? 0 : kind === 'small-network' ? 0.4 : 5,
    split: 'train',
  })
  const groups = graph.groups!
  model.layers.forEach((layer, layerIndex) => {
    const layerGroup = groups.find((group) => group.id === layer.id)!
    layerGroup.kind = 'layer'
    layerGroup.parentId = 'network'
    layerGroup.label = `${layerIndex === model.layers.length - 1 ? 'Output' : 'Hidden'} layer · ${layer.outputs} ${layer.outputs === 1 ? 'neuron' : 'neurons'}`
    for (let neuron = 0; neuron < layer.outputs; neuron++) {
      const prefix = `${layer.id}/neuron-${neuron}`
      const weightNodeIds = Array.from(
        { length: layer.inputs },
        (_, input) => `${layer.id}/weight-${input}-${neuron}`,
      )
      const biasNodeId = `${layer.id}/bias-${neuron}`
      const nodeIds = layerGroup.nodeIds.filter(
        (id) =>
          id.startsWith(`${prefix}/`) ||
          id === biasNodeId ||
          weightNodeIds.includes(id),
      )
      const inputNodeIds =
        layerIndex === 0
          ? Array.from({ length: layer.inputs }, (_, input) => `input-${input}`)
          : Array.from(
              { length: layer.inputs },
              (_, input) =>
                `${model.layers[layerIndex - 1].id}/neuron-${input}/activation`,
            )
      groups.push({
        id: prefix,
        label: `${layer.activation === 'relu' ? 'ReLU' : 'Linear'} neuron ${neuron + 1}`,
        kind: 'neuron',
        parentId: layer.id,
        nodeIds,
        position: { x: 450 + layerIndex * 1050, y: neuron * 400 },
        dimensions: { width: 200, height: 160 },
        detail: {
          inputNodeIds,
          weightNodeIds,
          biasNodeId,
          preNodeId: `${prefix}/sum`,
          outputNodeId: `${prefix}/activation`,
          activation: layer.activation,
        },
      })
    }
  })
  groups.push({
    id: 'network',
    label:
      kind === 'linear'
        ? 'Linear model'
        : kind === 'neuron'
          ? 'One ReLU neuron'
          : kind === 'small-network'
            ? 'MLP · 2 → 2 → 1'
            : 'MLP · 2 → 4 → 3 → 2',
    kind: 'network',
    nodeIds: groups
      .filter((group) => group.kind === 'layer')
      .flatMap((group) => group.nodeIds),
    position: { x: 260, y: -80 },
    dimensions: { width: 240, height: 180 },
  })
  if (kind === 'playground') {
    const builder = new ModelGraph()
    builder.graph = graph
    const last = model.layers.at(-1)!
    const parts = Array.from({ length: last.outputs }, (_, neuron) =>
      builder.node(
        `output-${neuron}`,
        'reshape',
        `Score ${neuron + 1}`,
        [`${last.id}/neuron-${neuron}/activation`],
        3600,
        neuron * 180,
        { shape: [1] },
      ),
    )
    const joined = builder.node(
      'output-vector',
      'concat',
      'Join class scores',
      parts,
      3830,
      0,
      { axis: 0, inputCount: parts.length },
    )
    const logits = builder.node(
      'network-logits',
      'reshape',
      'Class scores',
      [joined],
      4060,
      0,
      { shape: [1, 2] },
    )
    builder.node(
      'network-probabilities',
      'softmax',
      'Class probabilities',
      [logits],
      4310,
      -100,
    )
    builder.source(
      'target',
      'target',
      'Target class',
      value([1], [0]),
      4060,
      250,
    )
    builder.node(
      'loss',
      'cross-entropy',
      'Classification loss',
      [logits, 'target'],
      4310,
      220,
    )
  }
  graph.view = {
    expandedGroupIds: [
      'network',
      ...model.layers.map((layer) => layer.id),
      ...(kind === 'linear' || kind === 'neuron' ? ['layer-0/neuron-0'] : []),
    ],
    semanticZoom: true,
  }
  return graph
}

function probabilitiesPreset(): GraphModel {
  const b = new ModelGraph()
  b.source('scores', 'input', 'Scores', value([1, 4], [2, 1, 0.1, -1]), 80, 100)
  b.node(
    'probabilities',
    'softmax',
    'Softmax probabilities',
    ['scores'],
    380,
    100,
  )
  b.source('target', 'target', 'Target class', value([1], [0]), 380, 350)
  b.node(
    'loss',
    'cross-entropy',
    'Cross-entropy',
    ['scores', 'target'],
    710,
    200,
  )
  return b.graph
}

function attentionPreset(causal: boolean): GraphModel {
  const b = new ModelGraph()
  const q = b.source(
    'queries',
    'input',
    causal ? 'Queries · 3 tokens' : 'One query',
    value(causal ? [3, 2] : [1, 2], causal ? [1, 0, 0, 1, 1, 1] : [1, 0.5]),
    80,
    60,
  )
  const k = b.source(
    'keys',
    'input',
    'Keys · 3 tokens',
    value([3, 2], [1, 0, 0, 1, 1, 1]),
    80,
    340,
  )
  const v = b.source(
    'values',
    'input',
    'Values · 3 messages',
    value([3, 2], [1, 0, 0, 2, 1, 1]),
    80,
    620,
  )
  const start = b.graph.nodes.length
  const kt = b.node(
    'keys-transposed',
    'transpose',
    'Transpose keys',
    [k],
    360,
    340,
  )
  const dot = b.node(
    'query-key-products',
    'matmul',
    'Query · key scores',
    [q, kt],
    650,
    140,
  )
  const scale = b.source(
    'attention-scale',
    'input',
    '1 / √d',
    1 / Math.sqrt(2),
    650,
    400,
  )
  const scores = b.node(
    'attention-scores',
    'multiply',
    'Scaled scores',
    [dot, scale],
    920,
    160,
  )
  const masked = causal
    ? b.node(
        'masked-scores',
        'causal-mask',
        'Hide future tokens',
        [scores],
        1200,
        160,
      )
    : scores
  const weights = b.node(
    'attention-weights',
    'softmax',
    'Attention weights',
    [masked],
    causal ? 1480 : 1200,
    160,
  )
  b.node(
    'attention-output',
    'matmul',
    'Weighted messages',
    [weights, v],
    causal ? 1780 : 1500,
    160,
  )
  b.group(
    'attention',
    causal ? 'Causal self-attention' : 'One-query attention',
    'attention',
    start,
    420,
    100,
  )
  b.graph.view!.expandedGroupIds = ['attention']
  return b.graph
}

function decoderPreset(
  kind: 'embeddings' | 'block' | 'decoder',
  checkpointName: ModelCheckpoint,
): GraphModel {
  const b = new ModelGraph(),
    checkpoint = getCheckpoint(checkpointName),
    c = checkpoint.config
  b.graph.learningRate = 0.005
  const parameter = (id: string, label: string, x: number, y: number) =>
    b.source(
      id,
      id.toLowerCase().includes('bias') ? 'bias' : 'weight',
      label,
      structuredClone(checkpoint.parameters[id]),
      x,
      y,
    )
  const tokenIds = b.source(
    'token-ids',
    'input',
    'Token IDs',
    value([3], [0, 1, 2]),
    0,
    0,
  )
  const positionIds = b.source(
    'position-ids',
    'input',
    'Position IDs',
    value([3], [0, 1, 2]),
    0,
    280,
  )
  const embeddingStart = b.graph.nodes.length
  parameter('tokenEmbedding', 'Token embedding table · 5 × 8', 280, -240)
  const token = b.node(
    'token-vectors',
    'embedding',
    'Token vectors',
    ['tokenEmbedding', tokenIds],
    560,
    0,
  )
  parameter('positionEmbedding', 'Position embedding table · 12 × 8', 280, 520)
  const position = b.node(
    'position-vectors',
    'embedding',
    'Position vectors',
    ['positionEmbedding', positionIds],
    560,
    300,
  )
  let stream = b.node(
    'embedding-output',
    'add',
    'Token + position',
    [token, position],
    850,
    100,
  )
  b.group(
    'embeddings',
    'Token + position embeddings',
    'embedding',
    embeddingStart,
    350,
    100,
    undefined,
    {
      tokenIdsNodeId: tokenIds,
      positionIdsNodeId: positionIds,
      outputNodeId: stream,
      vocabulary: [...c.vocab],
      maxLength: c.maxLength,
      width: c.width,
      checkpoint: checkpointName,
    },
  )
  if (kind === 'embeddings') {
    b.graph.view!.expandedGroupIds = ['embeddings']
    return b.graph
  }

  const normalization = (
    input: string,
    prefix: string,
    x: number,
    y: number,
    parentId?: string,
  ) => {
    const start = b.graph.nodes.length
    parameter(`${prefix}.scale`, 'Learned scale γ', x, y - 220)
    parameter(`${prefix}.bias`, 'Learned shift β', x, y + 240)
    const output = b.node(
      `${prefix}.output`,
      'layer-norm',
      'Layer normalization',
      [input, `${prefix}.scale`, `${prefix}.bias`],
      x + 240,
      y,
    )
    b.group(
      prefix,
      'Layer normalization',
      'normalization',
      start,
      x,
      y,
      parentId,
    )
    return output
  }

  const dense = (
    input: string,
    prefix: string,
    biasId: string,
    activation: ActivationKind,
    label: string,
    x: number,
    y: number,
    parentId: string,
  ) => {
    const start = b.graph.nodes.length
    parameter(prefix, 'Weights', x, y - 220)
    const product = b.node(
      `${prefix}.product`,
      'matmul',
      'Weighted inputs',
      [input, prefix],
      x + 240,
      y,
    )
    parameter(biasId, 'Bias', x + 240, y + 260)
    const pre = b.node(
      `${prefix}.pre`,
      'add',
      'Weighted sum + bias',
      [product, biasId],
      x + 480,
      y,
    )
    const output = b.node(
      `${prefix}.output`,
      'activation',
      activation === 'relu' ? 'ReLU neurons' : 'Linear neurons',
      [pre],
      x + 720,
      y,
      { activation },
    )
    b.group(`${prefix}.layer`, label, 'layer', start, x, y, parentId, {
      inputNodeId: input,
      weightNodeId: prefix,
      biasNodeId: biasId,
      preNodeId: pre,
      outputNodeId: output,
      activation,
    })
    return output
  }

  const count = kind === 'block' ? 1 : c.blocks
  for (let block = 0; block < count; block++) {
    const prefix = `blocks.${block}`,
      blockStart = b.graph.nodes.length,
      blockInput = stream,
      x = 1300 + block * 5800
    const norm1 = normalization(stream, `${prefix}.norm1`, x, 100, prefix)
    const attentionStart = b.graph.nodes.length
    const projections = ['q', 'k', 'v'].map((name, index) => {
      const start = b.graph.nodes.length
      parameter(
        `${prefix}.${name}`,
        `${name.toUpperCase()} projection weights`,
        x + 520,
        index * 440 - 240,
      )
      const output = b.node(
        `${prefix}.${name}.output`,
        'matmul',
        name === 'q' ? 'Queries' : name === 'k' ? 'Keys' : 'Values',
        [norm1, `${prefix}.${name}`],
        x + 760,
        index * 440 - 200,
      )
      b.group(
        `${prefix}.${name}.projection`,
        name === 'q'
          ? 'Query projection'
          : name === 'k'
            ? 'Key projection'
            : 'Value projection',
        'projection',
        start,
        x + 560,
        index * 250 - 140,
        `${prefix}.attention`,
      )
      return output
    })
    const headWidth = c.width / c.heads
    const heads = Array.from({ length: c.heads }, (_, head) => {
      const headPrefix = `${prefix}.head.${head}`,
        start = b.graph.nodes.length,
        y = head * 800 - 280
      const [q, k, v] = projections.map((input, index) =>
        b.node(
          `${headPrefix}.${['q', 'k', 'v'][index]}`,
          'slice',
          `${['Query', 'Key', 'Value'][index]} coordinates`,
          [input],
          x + 1080,
          y + index * 200,
          { axis: 1, start: head * headWidth, end: (head + 1) * headWidth },
        ),
      )
      const kt = b.node(
        `${headPrefix}.kt`,
        'transpose',
        'Transpose keys',
        [k],
        x + 1320,
        y + 200,
      )
      const dot = b.node(
        `${headPrefix}.dot`,
        'matmul',
        'Query · key',
        [q, kt],
        x + 1560,
        y,
      )
      const scale = b.source(
        `${headPrefix}.scale`,
        'input',
        '1 / √head width',
        1 / Math.sqrt(headWidth),
        x + 1560,
        y + 250,
      )
      const scores = b.node(
        `${headPrefix}.scores`,
        'multiply',
        'Scaled scores',
        [dot, scale],
        x + 1800,
        y,
      )
      const mask = b.node(
        `${headPrefix}.mask`,
        'causal-mask',
        'Causal mask',
        [scores],
        x + 2040,
        y,
      )
      const weights = b.node(
        `${headPrefix}.weights`,
        'softmax',
        'Attention weights',
        [mask],
        x + 2280,
        y,
      )
      const output = b.node(
        `${headPrefix}.output`,
        'matmul',
        'Mix value vectors',
        [weights, v],
        x + 2520,
        y,
      )
      b.group(
        headPrefix,
        `Attention head ${head + 1} · width ${headWidth}`,
        'head',
        start,
        x + 1200,
        y,
        `${prefix}.attention`,
        {
          queryNodeId: q,
          keyNodeId: k,
          valueNodeId: v,
          scoreNodeId: scores,
          weightsNodeId: weights,
          outputNodeId: output,
        },
      )
      return output
    })
    const joined = b.node(
      `${prefix}.heads-concat`,
      'concat',
      'Concatenate heads',
      heads,
      x + 2820,
      100,
      { axis: 1, inputCount: c.heads },
    )
    parameter(`${prefix}.o`, 'Output projection weights', x + 2820, 420)
    const attention = b.node(
      `${prefix}.attention.output`,
      'matmul',
      'Attention output projection',
      [joined, `${prefix}.o`],
      x + 3080,
      100,
    )
    b.group(
      `${prefix}.attention`,
      'Multi-head causal attention · 2 heads',
      'attention',
      attentionStart,
      x + 700,
      100,
      prefix,
      {
        inputNodeId: norm1,
        queryNodeId: projections[0],
        keyNodeId: projections[1],
        valueNodeId: projections[2],
        outputNodeId: attention,
        heads: c.heads,
      },
    )
    const residual = b.node(
      `${prefix}.attention-residual`,
      'add',
      'Add residual',
      [blockInput, attention],
      x + 3340,
      100,
    )
    const norm2 = normalization(
      residual,
      `${prefix}.norm2`,
      x + 3620,
      100,
      prefix,
    )
    const mlpStart = b.graph.nodes.length
    const hidden = dense(
      norm2,
      `${prefix}.ff1`,
      `${prefix}.ff1Bias`,
      'relu',
      'Hidden layer · 8 → 16 · ReLU',
      x + 4020,
      -120,
      `${prefix}.mlp`,
    )
    const ffOutput = dense(
      hidden,
      `${prefix}.ff2`,
      `${prefix}.ff2Bias`,
      'identity',
      'Output layer · 16 → 8',
      x + 4860,
      -120,
      `${prefix}.mlp`,
    )
    b.group(
      `${prefix}.mlp`,
      'Feedforward MLP · 8 → 16 → 8',
      'mlp',
      mlpStart,
      x + 4080,
      100,
      prefix,
      {
        inputNodeId: norm2,
        outputNodeId: ffOutput,
        widths: [c.width, c.ffWidth, c.width],
      },
    )
    stream = b.node(
      `${prefix}.output`,
      'add',
      'Add residual',
      [residual, ffOutput],
      x + 5700,
      100,
    )
    b.group(
      prefix,
      `Transformer block ${block + 1}`,
      'transformer-block',
      blockStart,
      x,
      100,
      undefined,
      {
        inputNodeId: blockInput,
        outputNodeId: stream,
        width: c.width,
        heads: c.heads,
        ffWidth: c.ffWidth,
      },
    )
  }
  const outputX = 1400 + count * 5800
  const finalNorm = normalization(stream, 'finalNorm', outputX, 100)
  const outputStart = b.graph.nodes.length
  parameter('unembedding', 'Vocabulary projection · 8 × 5', outputX + 520, -160)
  const projection = b.node(
    'vocabulary-projection',
    'matmul',
    'Vocabulary scores',
    [finalNorm, 'unembedding'],
    outputX + 760,
    100,
  )
  parameter('outputBias', 'Vocabulary bias', outputX + 760, 380)
  const logits = b.node(
    'decoder-logits',
    'add',
    'Next-token logits',
    [projection, 'outputBias'],
    outputX + 1020,
    100,
  )
  b.group(
    'vocabulary',
    'Vocabulary projection',
    'projection',
    outputStart,
    outputX + 540,
    100,
  )
  b.node(
    'decoder-probabilities',
    'softmax',
    'Next-token probabilities',
    [logits],
    outputX + 1300,
    100,
  )
  b.source(
    'decoder-targets',
    'target',
    'Next-token targets',
    value([3], [1, 2, 3]),
    outputX + 1020,
    470,
  )
  b.node(
    'decoder-loss',
    'cross-entropy',
    'Next-token cross-entropy',
    [logits, 'decoder-targets'],
    outputX + 1300,
    420,
  )
  b.graph.view!.expandedGroupIds = []
  return b.graph
}

/** Keep named input/target cards as readable aliases, with actual data supplied
 * exclusively through dataset wires. Constants such as 1/√d remain model values. */
function attachPresetDataset(graph: GraphModel, dataset: DatasetKind, outputs: string[], datasetIndex = 0): GraphModel {
  const id = 'model-data'
  graph.nodes.unshift({ id, type: 'dataset', label: 'Dataset', position: { x: -320, y: 0 }, params: { dataset, datasetMode: 'sample', datasetIndex } })
  for (const [sourceSlot, target] of outputs.entries()) {
    if (!graph.nodes.some(node => node.id === target)) continue
    graph.edges.push({ id: `${id}→${target}`, source: id, sourceSlot, target, inputSlot: 0 })
  }
  return graph
}
