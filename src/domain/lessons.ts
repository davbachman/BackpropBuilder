import { createReluGateLessonGraph, createStarterGraph } from './examples'
import type { LessonDefinition } from './types'

export const lessons: LessonDefinition[] = [
  {
    id: 'one-neuron',
    name: 'Lesson 1: One neuron',
    prompt: 'A single scalar input flows through a weight, bias, sigmoid, and squared-error loss.',
    task: 'Run one full training step. Watch the prediction, loss, and parameter gradients appear.',
    completionCondition: 'Complete one forward, backward, and update sequence.',
    successMessage: 'You completed one training step and saw how the sigmoid neuron learns from error.',
    createGraph: createStarterGraph,
  },
  {
    id: 'relu-gate',
    name: 'Lesson 2: ReLU gate',
    prompt: 'This graph starts with a negative pre-activation so ReLU outputs zero.',
    task: 'Run the backward pass and observe the zero local derivative at the ReLU node.',
    completionCondition: 'Complete a training step with the ReLU graph.',
    successMessage: 'The ReLU gate blocked the upstream gradient because its input was negative.',
    createGraph: createReluGateLessonGraph,
  },
  {
    id: 'gradient-descent',
    name: 'Lesson 3: Gradient descent step',
    prompt: 'Repeated small parameter updates should move the scalar neuron toward the target.',
    task: 'Run 10 training steps and compare the loss before and after.',
    completionCondition: 'Run the 10-step training control.',
    successMessage: 'Repeated gradient descent updates reduced the loss for the starter neuron.',
    createGraph: createStarterGraph,
  },
]
