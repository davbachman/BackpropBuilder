import { mkdir, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

const root = fileURLToPath(new URL('../', import.meta.url))
const output = fileURLToPath(new URL('../public/models/', import.meta.url))
const server = await createServer({ root, server: { middlewareMode: true }, appType: 'custom', logLevel: 'error' })

try {
  const [{ LESSONS }, { createModelPreset }, { createStarterGraph }, { forwardPass, parameterValues }, { createProjectStateFile, parseProjectStateFile }] = await Promise.all([
    server.ssrLoadModule('/src/learning/presets.ts'),
    server.ssrLoadModule('/src/domain/modelPresets.ts'),
    server.ssrLoadModule('/src/domain/examples.ts'),
    server.ssrLoadModule('/src/domain/engine.ts'),
    server.ssrLoadModule('/src/domain/session.ts'),
  ])

  await mkdir(output, { recursive: true })
  for (const { id } of LESSONS) {
    await exportModel(id, createModelPreset(id))
  }
  await exportModel('block-untrained', createModelPreset('block', 'untrained'))
  await exportModel('decoder-untrained', createModelPreset('decoder', 'untrained'))
  await exportModel('starter-neuron', createStarterGraph(true))

  async function exportModel(filename, graph) {
    const evaluated = forwardPass(graph)
    const file = createProjectStateFile({
      graph: evaluated.graph,
      visualizationGraph: evaluated.graph,
      initialParameterValues: parameterValues(graph),
      selectedNodeIds: [],
      phase: 'edit',
      traceSteps: [],
      traceIndex: 0,
      epoch: 0,
      currentLoss: evaluated.loss ?? null,
      display: { showMath: true, showGradient: true, showCode: false, showVisualization: false },
    })
    file.savedAt = '2026-09-24T00:00:00.000Z'
    const json = JSON.stringify(file, null, 2) + '\n'
    const parsed = parseProjectStateFile(json)
    if (!parsed.ok) throw new Error(`${filename}: ${parsed.error}`)
    await writeFile(join(output, `${filename}.json`), json)
    process.stdout.write(`${filename}.json\n`)
  }
} finally {
  await server.close()
}
