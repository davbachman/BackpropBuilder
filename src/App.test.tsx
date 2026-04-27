import { act, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Position as FlowPosition } from '@xyflow/react'
import { describe, expect, it, vi } from 'vitest'
import appCss from './App.css?raw'
import builderEdgeSource from './components/BuilderEdge.tsx?raw'
import { BuilderEdge } from './components/BuilderEdge'
import { GraphCanvas } from './components/GraphCanvas'
import { DATASET_OPTIONS, MIN_NODE_HEIGHT, NODE_WIDTH, heightForInputCount } from './domain/engine'
import { scalarValue } from './domain/tensor'
import './index.css'
import App from './App'
import type { GraphModel } from './domain/types'

describe('Backprop Builder app', () => {
  it('renders the teaching workspace controls', () => {
    render(<App />)

    expect(screen.getByRole('heading', { name: /Backprop Builder/i })).toBeInTheDocument()
    expect(screen.getByText(/Node palette/i)).toBeInTheDocument()
    expect(screen.getByText(/Graph canvas/i)).toBeInTheDocument()
    expect(screen.getByText(/Inspector/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Load starter example/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Step$/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Step backward/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Step forward/i })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Run 10 training steps/i })).toBeInTheDocument()
  })

  it('activates the visualization panel on demand', async () => {
    const user = userEvent.setup()
    render(<App />)

    expect(screen.queryByRole('region', { name: /Visualization panel/i })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))
    await user.click(screen.getByRole('button', { name: /Show visualization/i }))

    expect(screen.getByRole('region', { name: /Visualization panel/i })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /Input-output visualization/i })).toBeInTheDocument()
  })

  it('starts with a blank canvas instead of the starter example', () => {
    render(<App />)

    expect(screen.getByText('0 nodes, 0 edges')).toBeInTheDocument()
    expect(screen.queryByText(/x = 2\.000/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/w = 0\.500/i)).not.toBeInTheDocument()
    expect(screen.getByText(/Add exactly one loss node/i)).toBeInTheDocument()
  })

  it('locks the workspace to the viewport and makes sidebars scroll internally', () => {
    const { container } = render(<App />)

    const shell = container.querySelector('.app-shell')
    const leftPanel = container.querySelector('.left-panel')
    const rightPanel = container.querySelector('.right-panel')
    const flowShell = container.querySelector('.flow-shell')

    expect(getComputedStyle(document.documentElement).height).toBe('100%')
    expect(getComputedStyle(document.body).height).toBe('100%')
    expect(getComputedStyle(document.body).overflow).toBe('hidden')
    expect(getComputedStyle(shell!).height).toBe('100vh')
    expect(getComputedStyle(shell!).overflow).toBe('hidden')
    expect(getComputedStyle(leftPanel!).overflowY).toBe('auto')
    expect(getComputedStyle(rightPanel!).overflowY).toBe('auto')
    expect(getComputedStyle(leftPanel!).minHeight).toBe('0px')
    expect(getComputedStyle(rightPanel!).minHeight).toBe('0px')
    expect(getComputedStyle(flowShell!).minHeight).toBe('0px')
  })

  it('loads the starter graph and runs a full training step', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))
    expect(screen.getByText(/x = 2/i)).toBeInTheDocument()
    expect(screen.getByText(/w = 0.500/i)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /Run one full training step/i }))
    expect(screen.getByText(/Epoch 1/i)).toBeInTheDocument()
    expect(screen.getByText(/Current loss/i)).toBeInTheDocument()
  })

  it('uses the slowest playback delay at the left edge of the speed slider', async () => {
    const user = userEvent.setup()
    const setTimeoutSpy = vi.spyOn(window, 'setTimeout')

    try {
      render(<App />)

      await user.click(screen.getByRole('button', { name: /Load starter example/i }))

      const speedSlider = screen.getByLabelText('Speed')
      fireEvent.change(speedSlider, { target: { value: '250' } })

      await user.click(screen.getByRole('button', { name: 'Play' }))

      expect(setTimeoutSpy).toHaveBeenCalledWith(expect.any(Function), 1800)
    } finally {
      setTimeoutSpy.mockRestore()
    }
  })

  it('updates visualization predictions when Play reaches a completed forward pass', async () => {
    vi.useFakeTimers()

    try {
      render(<App />)

      fireEvent.click(screen.getByRole('button', { name: /Load starter example/i }))
      fireEvent.click(screen.getByRole('button', { name: /Show visualization/i }))
      const initialPrediction = visualizationPredictionPath()

      fireEvent.change(screen.getByDisplayValue('0.5'), { target: { value: '1' } })
      expect(visualizationPredictionPath()).toBe(initialPrediction)

      fireEvent.click(screen.getByRole('button', { name: 'Play' }))
      for (let index = 0; index < 4; index += 1) {
        await act(async () => {
          await vi.advanceTimersByTimeAsync(900)
        })
      }

      expect(visualizationPredictionPath()).not.toBe(initialPrediction)
    } finally {
      vi.useRealTimers()
    }
  })

  it('shows connected variable names in starter graph node formulas', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))

    expect(screen.getByText('z1 = x * w')).toBeInTheDocument()
    expect(screen.getByText('z2 = z1 + b')).toBeInTheDocument()
    expect(screen.getAllByText('z3 = sigmoid(z2)').length).toBeGreaterThan(0)
    expect(screen.getByText('L = 0.5 * (z3 - y)^2')).toBeInTheDocument()
  })

  it('renders a loss dropdown and updates the displayed loss formula', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        { id: 'pred', type: 'input', label: 'pred', position: { x: 80, y: 80 }, params: { value: scalarValue(0.4) } },
        { id: 'target', type: 'target', label: 'y', position: { x: 80, y: 240 }, params: { value: scalarValue(1) } },
        { id: 'loss', type: 'loss', label: 'loss', position: { x: 340, y: 160 }, params: {} },
      ],
      edges: [
        { id: 'pred-loss', source: 'pred', target: 'loss', inputSlot: 0 },
        { id: 'target-loss', source: 'target', target: 'loss', inputSlot: 1 },
      ],
    }
    const onLossChange = vi.fn()
    const noop = vi.fn()
    const { rerender } = render(
      <GraphCanvas
        graph={graph}
        showMath
        showGradient
        phase="edit"
        onGraphChange={noop}
        onSelectionChange={noop}
        onCreateNode={noop}
        onCancelPendingPlacement={noop}
        onNodeValueChange={noop}
        onActivationChange={noop}
        onLossChange={onLossChange}
        onGroupCreate={noop}
        onGroupExplode={noop}
        onGroupMove={noop}
      />,
    )
    const lossSelect = screen
      .getAllByRole('combobox', { hidden: true })
      .find((element) => element.getAttribute('aria-label') === 'loss')
    expect(lossSelect).toBeDefined()
    expect(lossSelect).toHaveValue('squared-error')

    fireEvent.change(lossSelect!, { target: { value: 'mse' } })

    expect(onLossChange).toHaveBeenCalledWith('loss', 'mse')

    rerender(
      <GraphCanvas
        graph={{
          ...graph,
          nodes: graph.nodes.map((node) =>
            node.id === 'loss' ? { ...node, params: { ...node.params, loss: 'mse' } } : node,
          ),
        }}
        showMath
        showGradient
        phase="edit"
        onGraphChange={noop}
        onSelectionChange={noop}
        onCreateNode={noop}
        onCancelPendingPlacement={noop}
        onNodeValueChange={noop}
        onActivationChange={noop}
        onLossChange={onLossChange}
        onGroupCreate={noop}
        onGroupExplode={noop}
        onGroupMove={noop}
      />,
    )

    expect(screen.getByText('L = (pred - y)^2')).toBeInTheDocument()
  })

  it('renders a dataset dropdown with toy dataset choices', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        {
          id: 'dataset',
          type: 'dataset',
          label: 'dataset',
          position: { x: 80, y: 80 },
          params: { dataset: 'line-1d' },
        },
      ],
      edges: [],
    }
    const onDatasetChange = vi.fn()
    const noop = vi.fn()
    render(
      <GraphCanvas
        graph={graph}
        showMath
        showGradient
        phase="edit"
        onGraphChange={noop}
        onSelectionChange={noop}
        onCreateNode={noop}
        onCancelPendingPlacement={noop}
        onNodeValueChange={noop}
        onActivationChange={noop}
        onLossChange={noop}
        onDatasetChange={onDatasetChange}
        onGroupCreate={noop}
        onGroupExplode={noop}
        onGroupMove={noop}
      />,
    )

    const datasetSelect = screen
      .getAllByRole('combobox', { hidden: true })
      .find((element) => element.getAttribute('aria-label') === 'dataset')
    expect(datasetSelect).toBeDefined()
    expect(datasetSelect).toHaveValue('line-1d')
    expect(Array.from(datasetSelect!.querySelectorAll('option')).map((option) => option.textContent)).toEqual(
      DATASET_OPTIONS.map((option) => option.label),
    )

    fireEvent.change(datasetSelect!, { target: { value: 'circle-center' } })

    expect(onDatasetChange).toHaveBeenCalledWith('dataset', 'circle-center')
  })

  it('hides value editors on input and target nodes fed by a dataset', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        {
          id: 'dataset',
          type: 'dataset',
          label: 'dataset',
          position: { x: 40, y: 80 },
          params: { dataset: 'line-1d' },
        },
        {
          id: 'x',
          type: 'input',
          label: 'x',
          position: { x: 260, y: 80 },
          params: { value: scalarValue(0) },
        },
        {
          id: 'target',
          type: 'target',
          label: 'y',
          position: { x: 260, y: 240 },
          params: { value: scalarValue(0) },
        },
        {
          id: 'manual-x',
          type: 'input',
          label: 'manual x',
          position: { x: 260, y: 400 },
          params: { value: scalarValue(4) },
        },
        {
          id: 'manual-target',
          type: 'target',
          label: 'manual y',
          position: { x: 260, y: 560 },
          params: { value: scalarValue(5) },
        },
      ],
      edges: [
        { id: 'dataset-x', source: 'dataset', sourceSlot: 0, target: 'x', inputSlot: 0 },
        { id: 'dataset-target', source: 'dataset', sourceSlot: 1, target: 'target', inputSlot: 0 },
      ],
    }
    const noop = vi.fn()
    render(
      <GraphCanvas
        graph={graph}
        showMath
        showGradient
        phase="edit"
        onGraphChange={noop}
        onSelectionChange={noop}
        onCreateNode={noop}
        onCancelPendingPlacement={noop}
        onNodeValueChange={noop}
        onActivationChange={noop}
        onLossChange={noop}
        onDatasetChange={noop}
        onGroupCreate={noop}
        onGroupExplode={noop}
        onGroupMove={noop}
      />,
    )

    expect(screen.queryAllByDisplayValue('0')).toHaveLength(0)
    expect(screen.getByDisplayValue('4')).toBeInTheDocument()
    expect(screen.getByDisplayValue('5')).toBeInTheDocument()
  })

  it('accepts tensor literals in source nodes and keeps node tensor displays compact with full hover text', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))

    const xInput = screen.getByDisplayValue('2')
    fireEvent.change(xInput, { target: { value: '[1,2,3]' } })

    expect(screen.getByText('x = [1.000,...]')).toHaveAttribute(
      'data-tooltip',
      'x = [3] [1.000, 2.000, 3.000]',
    )
    expect(screen.getByText('out [1.000,...]')).toHaveAttribute(
      'data-tooltip',
      'out [3] [1.000, 2.000, 3.000]',
    )
  })

  it('starts visible forward stepping at the first computation after source values', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))
    await user.click(screen.getByRole('button', { name: /^Step$/i }))

    expect(screen.getByRole('heading', { name: 'Evaluate x * w' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Evaluate x' })).not.toBeInTheDocument()
  })

  it('reveals forward values only after their computation step is reached', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))
    await user.click(screen.getByRole('button', { name: /^Step$/i }))

    expect(screen.getByRole('heading', { name: 'Evaluate x * w' })).toBeInTheDocument()
    expect(screen.queryByText('out 0.700')).not.toBeInTheDocument()
    expect(screen.queryByText('out 0.668')).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /^Step$/i }))

    expect(screen.getByRole('heading', { name: 'Evaluate xw + b' })).toBeInTheDocument()
    expect(screen.getByText('out 0.700')).toBeInTheDocument()
    expect(screen.queryByText('out 0.668')).not.toBeInTheDocument()
  })

  it('reveals backward gradients only after their backprop step is reached', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))
    for (let index = 0; index < 5; index += 1) {
      await user.click(screen.getByRole('button', { name: /^Step$/i }))
    }

    expect(screen.getByRole('heading', { name: 'Backpropagate through loss' })).toBeInTheDocument()
    expect(screen.queryByText(/grad -0\.147/)).not.toBeInTheDocument()

    for (let index = 0; index < 3; index += 1) {
      await user.click(screen.getByRole('button', { name: /^Step$/i }))
    }

    expect(screen.getByRole('heading', { name: 'Backpropagate through x * w' })).toBeInTheDocument()
    expect(screen.getByText(/grad -0\.147/)).toBeInTheDocument()
  })

  it('does not render floating edge value bubbles on the canvas', async () => {
    const user = userEvent.setup()
    const { container } = render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))
    await user.click(screen.getByRole('button', { name: /^Step$/i }))

    expect(screen.getByRole('heading', { name: 'Evaluate x * w' })).toBeInTheDocument()
    expect(container.querySelectorAll('.edge-label')).toHaveLength(0)

    for (let index = 0; index < 4; index += 1) {
      await user.click(screen.getByRole('button', { name: /^Step$/i }))
    }

    expect(screen.getByRole('heading', { name: 'Backpropagate through loss' })).toBeInTheDocument()
    expect(container.querySelectorAll('.edge-label')).toHaveLength(0)
  })

  it('does not create React Flow edge label bubbles', () => {
    expect(builderEdgeSource).not.toMatch(/EdgeLabelRenderer/)
    expect(builderEdgeSource).not.toMatch(/edge-label/)
  })

  it('adds a selected class to selected wire paths', () => {
    const { container } = render(
      <svg>
        <BuilderEdge
          id="selected-edge"
          source="x"
          target="mul"
          selected
          sourceX={0}
          sourceY={0}
          targetX={120}
          targetY={40}
          sourcePosition={FlowPosition.Right}
          targetPosition={FlowPosition.Left}
          data={{ showGradient: true, active: false, phase: 'edit' }}
        />
      </svg>,
    )

    expect(container.querySelector('path.builder-edge')).toHaveClass('is-selected')
  })

  it('moves from the last real backward computation to parameter updates instead of leaf inputs', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))
    for (let index = 0; index < 8; index += 1) {
      await user.click(screen.getByRole('button', { name: /^Step$/i }))
    }

    expect(screen.getByRole('heading', { name: 'Backpropagate through x * w' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /^Step$/i }))

    expect(screen.getByRole('heading', { name: 'Update w' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /Backpropagate through x|Backpropagate through y/ })).not.toBeInTheDocument()
  })

  it('uses palette selection as a one-shot canvas placement tool', async () => {
    const user = userEvent.setup()
    const { container } = render(<App />)

    await user.click(screen.getByRole('button', { name: /^Weight$/i }))
    expect(screen.getByText('0 nodes, 0 edges')).toBeInTheDocument()
    expect(screen.getByText(/Click the graph canvas to place Weight/i)).toBeInTheDocument()

    const pane = container.querySelector('.react-flow__pane')
    expect(pane).toBeInstanceOf(HTMLElement)
    fireEvent.click(pane!, { clientX: 480, clientY: 260 })

    expect(screen.getByText('1 nodes, 0 edges')).toBeInTheDocument()
    expect(screen.queryByText(/Click the graph canvas to place Weight/i)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /^Multiply$/i }))
    expect(screen.getByText('1 nodes, 0 edges')).toBeInTheDocument()
    expect(screen.getByText(/Click the graph canvas to place Multiply/i)).toBeInTheDocument()

    fireEvent.click(pane!, { clientX: 620, clientY: 320 })
    expect(screen.getByText('2 nodes, 0 edges')).toBeInTheDocument()
    expect(screen.queryByText(/Click the graph canvas to place Multiply/i)).not.toBeInTheDocument()
  })

  it('undoes graph edits repeatedly with Command-Z', async () => {
    const user = userEvent.setup()
    const { container } = render(<App />)
    const pane = container.querySelector('.react-flow__pane')
    expect(pane).toBeInstanceOf(HTMLElement)

    await user.click(screen.getByRole('button', { name: /^Weight$/i }))
    fireEvent.click(pane!, { clientX: 480, clientY: 260 })
    await user.click(screen.getByRole('button', { name: /^Input$/i }))
    fireEvent.click(pane!, { clientX: 300, clientY: 260 })

    expect(screen.getByText('2 nodes, 0 edges')).toBeInTheDocument()

    fireEvent.keyDown(document, { key: 'z', metaKey: true })
    expect(screen.getByText('1 nodes, 0 edges')).toBeInTheDocument()

    fireEvent.keyDown(document, { key: 'z', metaKey: true })
    expect(screen.getByText('0 nodes, 0 edges')).toBeInTheDocument()
  })

  it('copies and pastes the selected graph node with Command-C and Command-V', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))

    fireEvent.keyDown(document, { key: 'c', metaKey: true })
    fireEvent.keyDown(document, { key: 'v', metaKey: true })

    expect(screen.getByText('9 nodes, 7 edges')).toBeInTheDocument()

    fireEvent.keyDown(document, { key: 'z', metaKey: true })

    expect(screen.getByText('8 nodes, 7 edges')).toBeInTheDocument()
  })

  it('undoes source value edits with Command-Z', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))
    const xInput = screen.getByDisplayValue('2')
    fireEvent.change(xInput, { target: { value: '[1,2,3]' } })

    expect(screen.getByText('x = [1.000,...]')).toBeInTheDocument()

    fireEvent.keyDown(document, { key: 'z', metaKey: true })

    expect(screen.getByText('x = 2.000')).toBeInTheDocument()
  })

  it('undoes a full training step with Command-Z', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))
    await user.click(screen.getByRole('button', { name: /Run one full training step/i }))

    expect(screen.getByText('Epoch 1')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Update w' })).toBeInTheDocument()

    fireEvent.keyDown(document, { key: 'z', metaKey: true })

    expect(screen.getByText('Epoch 0')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Ready to evaluate' })).toBeInTheDocument()
  })

  it('numbers palette-created node names independently by node type', async () => {
    const user = userEvent.setup()
    const { container } = render(<App />)
    const pane = container.querySelector('.react-flow__pane')
    expect(pane).toBeInstanceOf(HTMLElement)

    await user.click(screen.getByRole('button', { name: /^Input$/i }))
    fireEvent.click(pane!, { clientX: 440, clientY: 220 })
    await user.click(screen.getByRole('button', { name: /^Input$/i }))
    fireEvent.click(pane!, { clientX: 620, clientY: 220 })
    await user.click(screen.getByRole('button', { name: /^Weight$/i }))
    fireEvent.click(pane!, { clientX: 440, clientY: 380 })
    await user.click(screen.getByRole('button', { name: /^Target$/i }))
    fireEvent.click(pane!, { clientX: 620, clientY: 380 })

    const nodeTitles = Array.from(container.querySelectorAll('.builder-node .node-title-row strong')).map((element) =>
      element.textContent?.trim(),
    )

    expect(nodeTitles).toEqual(expect.arrayContaining(['x1', 'x2', 'w1', 'y1']))
    expect(nodeTitles).not.toEqual(expect.arrayContaining(['w3', 'y4']))
  })

  it('clears selection and pending placement after deleting a newly placed palette node', async () => {
    const user = userEvent.setup()
    const { container } = render(<App />)

    await user.click(screen.getByRole('button', { name: /^Weight$/i }))
    const pane = container.querySelector('.react-flow__pane')
    expect(pane).toBeInstanceOf(HTMLElement)
    fireEvent.click(pane!, { clientX: 480, clientY: 260 })

    expect(screen.getByText('1 nodes, 0 edges')).toBeInTheDocument()

    await user.keyboard('{Backspace}')
    expect(screen.getByText('0 nodes, 0 edges')).toBeInTheDocument()

    fireEvent.click(pane!, { clientX: 620, clientY: 320 })
    expect(screen.getByText('0 nodes, 0 edges')).toBeInTheDocument()
    expect(screen.queryByText(/Click the graph canvas to place Weight/i)).not.toBeInTheDocument()
  })

  it('cancels pending placement when the user interacts with an existing canvas element', async () => {
    const user = userEvent.setup()
    const { container } = render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))
    await user.click(screen.getByRole('button', { name: /^Weight$/i }))
    expect(screen.getByText(/Click the graph canvas to place Weight/i)).toBeInTheDocument()

    fireEvent.pointerDown(screen.getByText('x'))
    expect(screen.queryByText(/Click the graph canvas to place Weight/i)).not.toBeInTheDocument()

    const pane = container.querySelector('.react-flow__pane')
    expect(pane).toBeInstanceOf(HTMLElement)
    fireEvent.click(pane!, { clientX: 600, clientY: 360 })

    expect(screen.getByText('8 nodes, 7 edges')).toBeInTheDocument()
  })

  it('keeps editable controls inside graph nodes out of React Flow drag and wheel gestures', async () => {
    const user = userEvent.setup()
    const { container } = render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))

    const nodeControls = container.querySelectorAll('.react-flow__node input, .react-flow__node select')
    expect(nodeControls.length).toBeGreaterThan(0)

    nodeControls.forEach((control) => {
      expect(control).toHaveClass('nodrag')
      expect(control).toHaveClass('nowheel')
    })
  })

  it('renders flexible add input handles and a left-edge add input control', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        {
          id: 'add-1',
          type: 'add',
          label: 'add',
          position: { x: 120, y: 80 },
          dimensions: { height: heightForInputCount(4) },
          params: { inputCount: 4 },
        },
      ],
      edges: [],
    }

    const { container } = render(
      <GraphCanvas
        graph={graph}
        showMath
        showGradient
        phase="edit"
        onGraphChange={vi.fn()}
        onSelectionChange={vi.fn()}
        onCreateNode={vi.fn()}
        onCancelPendingPlacement={vi.fn()}
        onNodeValueChange={vi.fn()}
        onActivationChange={vi.fn()}
        onGroupCreate={vi.fn()}
        onGroupExplode={vi.fn()}
        onGroupMove={vi.fn()}
      />,
    )

    const addNode = container.querySelector('.node-add')
    expect(addNode).toBeInTheDocument()
    expect(addNode?.querySelectorAll('.node-handle.target')).toHaveLength(4)
    expect(addNode?.querySelector('.node-bottom-resize')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Add input to add/i })).toHaveClass('node-add-input-button')
  })

  it('adds a third flexible input without growing a two-input add node', async () => {
    const user = userEvent.setup()
    const onGraphChange = vi.fn()
    const initialHeight = heightForInputCount(2)
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        {
          id: 'add-1',
          type: 'add',
          label: 'add',
          position: { x: 120, y: 80 },
          dimensions: { height: initialHeight },
          params: { inputCount: 2 },
        },
      ],
      edges: [],
    }

    render(
      <GraphCanvas
        graph={graph}
        showMath
        showGradient
        phase="edit"
        onGraphChange={onGraphChange}
        onSelectionChange={vi.fn()}
        onCreateNode={vi.fn()}
        onCancelPendingPlacement={vi.fn()}
        onNodeValueChange={vi.fn()}
        onActivationChange={vi.fn()}
        onGroupCreate={vi.fn()}
        onGroupExplode={vi.fn()}
        onGroupMove={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: /Add input to add/i }))

    expect(onGraphChange).toHaveBeenCalledWith(
      expect.objectContaining({
        nodes: [
          expect.objectContaining({
            id: 'add-1',
            dimensions: expect.objectContaining({ height: initialHeight }),
            params: expect.objectContaining({ inputCount: 3 }),
          }),
        ],
      }),
    )
  })

  it('shows a merge action for multiple selected graph nodes', async () => {
    const user = userEvent.setup()
    const onGroupCreate = vi.fn()
    const graph: GraphModel = {
      learningRate: 0.1,
      nodes: [
        {
          id: 'input-1',
          type: 'input',
          label: 'x1',
          position: { x: 80, y: 80 },
          params: { value: 1 },
        },
        {
          id: 'weight-1',
          type: 'weight',
          label: 'w1',
          position: { x: 80, y: 240 },
          params: { value: 0.5 },
        },
      ],
      edges: [],
    }

    render(
      <GraphCanvas
        graph={graph}
        selectedNodeIds={['input-1', 'weight-1']}
        showMath
        showGradient
        phase="edit"
        onGraphChange={vi.fn()}
        onSelectionChange={vi.fn()}
        onCreateNode={vi.fn()}
        onCancelPendingPlacement={vi.fn()}
        onNodeValueChange={vi.fn()}
        onActivationChange={vi.fn()}
        onGroupCreate={onGroupCreate}
        onGroupExplode={vi.fn()}
        onGroupMove={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: /Merge selection/i }))

    expect(onGroupCreate).toHaveBeenCalledOnce()
  })

  it('renders a merged visual group as one selectable node with an explode action', async () => {
    const user = userEvent.setup()
    const onGroupExplode = vi.fn()
    const graph: GraphModel = {
      learningRate: 0.1,
      groups: [
        {
          id: 'group-1',
          label: 'Group 1',
          nodeIds: ['input-1', 'weight-1'],
          position: { x: 60, y: 60 },
          dimensions: { width: NODE_WIDTH, height: MIN_NODE_HEIGHT },
        },
      ],
      nodes: [
        {
          id: 'input-1',
          type: 'input',
          label: 'x1',
          position: { x: 80, y: 80 },
          params: { value: 1 },
        },
        {
          id: 'weight-1',
          type: 'weight',
          label: 'w1',
          position: { x: 80, y: 240 },
          params: { value: 0.5 },
        },
      ],
      edges: [],
    }

    const { container } = render(
      <GraphCanvas
        graph={graph}
        selectedGroupId="group-1"
        showMath
        showGradient
        phase="edit"
        onGraphChange={vi.fn()}
        onSelectionChange={vi.fn()}
        onCreateNode={vi.fn()}
        onCancelPendingPlacement={vi.fn()}
        onNodeValueChange={vi.fn()}
        onActivationChange={vi.fn()}
        onGroupCreate={vi.fn()}
        onGroupExplode={onGroupExplode}
        onGroupMove={vi.fn()}
      />,
    )

    expect(screen.getByText('Group 1')).toBeInTheDocument()
    expect(container.querySelectorAll('.builder-node')).toHaveLength(0)
    expect(container.querySelector('.visual-group-node')).toHaveStyle({
      width: `${NODE_WIDTH}px`,
      height: `${MIN_NODE_HEIGHT}px`,
    })
    expect(container.querySelectorAll('.visual-group-node .source-handle.group-handle')).toHaveLength(2)

    await user.click(screen.getByRole('button', { name: /Explode group/i }))

    expect(onGroupExplode).toHaveBeenCalledWith('group-1')
  })

  it('renders one group handle for each edge entering and leaving a merged visual group', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      groups: [
        {
          id: 'group-1',
          label: 'Group 1',
          nodeIds: ['mul', 'add'],
          position: { x: 300, y: 150 },
          dimensions: { width: NODE_WIDTH, height: MIN_NODE_HEIGHT },
        },
      ],
      nodes: [
        { id: 'x', type: 'input', label: 'x', position: { x: 40, y: 60 }, params: { value: 1 } },
        { id: 'w', type: 'weight', label: 'w', position: { x: 40, y: 260 }, params: { value: 0.5 } },
        { id: 'b', type: 'bias', label: 'b', position: { x: 320, y: 360 }, params: { value: 0 } },
        { id: 'mul', type: 'multiply', label: 'x * w', position: { x: 320, y: 160 }, params: {} },
        { id: 'add', type: 'add', label: 'xw + b', position: { x: 600, y: 260 }, params: {} },
        { id: 'pred', type: 'activation', label: 'activation', position: { x: 880, y: 160 }, params: { activation: 'sigmoid' } },
      ],
      edges: [
        { id: 'x-mul', source: 'x', target: 'mul', inputSlot: 0 },
        { id: 'w-mul', source: 'w', target: 'mul', inputSlot: 1 },
        { id: 'mul-add', source: 'mul', target: 'add', inputSlot: 0 },
        { id: 'b-add', source: 'b', target: 'add', inputSlot: 1 },
        { id: 'add-act', source: 'add', target: 'pred', inputSlot: 0 },
      ],
    }

    const { container } = render(
      <GraphCanvas
        graph={graph}
        showMath
        showGradient
        phase="edit"
        onGraphChange={vi.fn()}
        onSelectionChange={vi.fn()}
        onCreateNode={vi.fn()}
        onCancelPendingPlacement={vi.fn()}
        onNodeValueChange={vi.fn()}
        onActivationChange={vi.fn()}
        onGroupCreate={vi.fn()}
        onGroupExplode={vi.fn()}
        onGroupMove={vi.fn()}
      />,
    )

    const groupNode = container.querySelector('.visual-group-node')
    expect(groupNode?.querySelectorAll('.group-handle.target')).toHaveLength(3)
    expect(groupNode?.querySelectorAll('.group-handle.source')).toHaveLength(1)
    expect(container.querySelector('.react-flow__edge[data-id="mul-add"]')).not.toBeInTheDocument()
  })

  it('shows output and gradient metrics for merged visual group boundary outputs', () => {
    const graph: GraphModel = {
      learningRate: 0.1,
      groups: [
        {
          id: 'group-1',
          label: 'Group 1',
          nodeIds: ['mul', 'add'],
          position: { x: 300, y: 150 },
          dimensions: { width: NODE_WIDTH, height: MIN_NODE_HEIGHT },
        },
      ],
      nodes: [
        { id: 'x', type: 'input', label: 'x', position: { x: 40, y: 60 }, params: { value: 1 } },
        { id: 'w', type: 'weight', label: 'w', position: { x: 40, y: 260 }, params: { value: 0.5 } },
        { id: 'mul', type: 'multiply', label: 'x * w', position: { x: 320, y: 160 }, params: {} },
        { id: 'add', type: 'add', label: 'xw + b', position: { x: 600, y: 260 }, params: {} },
        { id: 'pred', type: 'activation', label: 'activation', position: { x: 880, y: 160 }, params: { activation: 'sigmoid' } },
      ],
      edges: [
        { id: 'x-mul', source: 'x', target: 'mul', inputSlot: 0 },
        { id: 'w-mul', source: 'w', target: 'mul', inputSlot: 1 },
        { id: 'mul-add', source: 'mul', target: 'add', inputSlot: 0 },
        {
          id: 'add-act',
          source: 'add',
          target: 'pred',
          inputSlot: 0,
          value: scalarValue(0.7),
          grad: scalarValue(-0.2),
        },
      ],
    }

    render(
      <GraphCanvas
        graph={graph}
        showMath
        showGradient
        phase="backward"
        onGraphChange={vi.fn()}
        onSelectionChange={vi.fn()}
        onCreateNode={vi.fn()}
        onCancelPendingPlacement={vi.fn()}
        onNodeValueChange={vi.fn()}
        onActivationChange={vi.fn()}
        onGroupCreate={vi.fn()}
        onGroupExplode={vi.fn()}
        onGroupMove={vi.fn()}
      />,
    )

    expect(screen.getByText('out 0.700')).toBeInTheDocument()
    expect(screen.getByText('grad -0.200')).toBeInTheDocument()
  })

  it('does not repeat a node type when the node title already says it', async () => {
    const user = userEvent.setup()
    const { container } = render(<App />)
    const pane = container.querySelector('.react-flow__pane')
    expect(pane).toBeInstanceOf(HTMLElement)

    await user.click(screen.getByRole('button', { name: /^Multiply$/i }))
    fireEvent.click(pane!, { clientX: 480, clientY: 260 })
    await user.click(screen.getByRole('button', { name: /^Activation$/i }))
    fireEvent.click(pane!, { clientX: 660, clientY: 260 })

    const duplicateHeaders = Array.from(container.querySelectorAll('.builder-node')).filter((node) => {
      const title = node.querySelector('.node-title-row strong')?.textContent?.trim().toLowerCase()
      const kind = node.querySelector('.node-kind')?.textContent?.trim().toLowerCase()
      return title && kind && title === kind
    })

    expect(duplicateHeaders).toHaveLength(0)
  })

  it('does not duplicate selected node details in the right sidebar', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))

    expect(screen.queryByText('Selected node')).not.toBeInTheDocument()
    expect(screen.getByText(/Formula/i)).toBeInTheDocument()
  })

  it('highlights the selected graph node with the selected highlight class', async () => {
    const user = userEvent.setup()
    const { container } = render(<App />)

    await user.click(screen.getByRole('button', { name: /Load starter example/i }))

    const xTitle = Array.from(container.querySelectorAll('.builder-node .node-title-row strong')).find(
      (element) => element.textContent?.trim() === 'x',
    )
    expect(xTitle).toBeInstanceOf(HTMLElement)
    fireEvent.click(xTitle!)

    const selectedNodes = container.querySelectorAll('.builder-node.is-selected')
    const selectedNode = selectedNodes.item(0) as HTMLElement
    expect(selectedNodes).toHaveLength(1)
    expect(selectedNode).toContainElement(xTitle as HTMLElement)
    expect(selectedNode).not.toHaveClass('is-active')
  })

  it('reverses the dash animation for backward edges', () => {
    expect(appCss).toMatch(/\.react-flow__edge\.animated\s+path\.builder-edge\.is-backward\s*{[^}]*animation-direction:\s*reverse;/)
  })

  it('records lesson actions and prepares a JSON summary download', async () => {
    const createObjectURL = vi.fn(() => 'blob:summary')
    const revokeObjectURL = vi.fn()
    vi.stubGlobal('URL', { createObjectURL, revokeObjectURL })
    const click = vi.fn()
    const originalCreateElement = document.createElement.bind(document)
    vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      const element = originalCreateElement(tagName)
      if (tagName === 'a') {
        Object.defineProperty(element, 'click', { value: click })
      }
      return element
    })

    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /Lesson drawer/i }))
    await user.click(screen.getByRole('button', { name: /Start Lesson 1/i }))
    await user.click(screen.getByRole('button', { name: /Run one full training step/i }))
    await user.click(screen.getByRole('button', { name: /Download session summary/i }))

    expect(createObjectURL).toHaveBeenCalled()
    expect(click).toHaveBeenCalled()
  })
})

function visualizationPredictionPath(): string | null | undefined {
  return document.querySelector('.visualization-prediction-line')?.getAttribute('d')
}
