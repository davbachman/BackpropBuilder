import { parameterValues } from './engine'
import type { GraphModel, TensorValue, TrainingSessionSummary } from './types'

export function createSessionSummary(input: {
  lessonName: string
  graph: GraphModel
  initialParameterValues: Record<string, TensorValue>
  trainingSteps: number
  finalLoss: number | null
  completedLessonActions: string[]
}): TrainingSessionSummary {
  return {
    timestamp: new Date().toISOString(),
    lessonName: input.lessonName,
    graph: {
      nodes: input.graph.nodes,
      edges: input.graph.edges,
    },
    initialParameterValues: input.initialParameterValues,
    finalParameterValues: parameterValues(input.graph),
    learningRate: input.graph.learningRate,
    trainingSteps: input.trainingSteps,
    finalLoss: input.finalLoss,
    completedLessonActions: input.completedLessonActions,
  }
}

export function downloadSessionSummary(summary: TrainingSessionSummary): void {
  const blob = new Blob([JSON.stringify(summary, null, 2)], {
    type: 'application/json',
  })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `backprop-builder-summary-${summary.timestamp.slice(0, 10)}.json`
  anchor.click()
  URL.revokeObjectURL(url)
}
