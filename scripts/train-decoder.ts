/** Reproduce the included checkpoints: node --experimental-strip-types scripts/train-decoder.ts
 * Seed 7, full determinism, 360 Adam updates, fixed synthetic corpus and budget.
 * No network, Python, external model weights, or additional dependencies. */
import { writeFileSync, mkdirSync } from 'node:fs'
import { createDecoder, decoderLoss, exportCheckpoint, sequenceExamples, trainDecoderStep, createAdamState, seededRandom } from '../src/learning/decoder.ts'
import type { CheckpointMetadata } from '../src/learning/decoder.ts'

const seed = 7, steps = 360, learningRate = 0.01
const model = createDecoder(seed), state = createAdamState(), examples = sequenceExamples('training'), random = seededRandom(seed + 1)
const initialTraining = decoderLoss(model, 'training'), initialValidation = decoderLoss(model, 'validation')
const metadata: CheckpointMetadata = {
  name: 'Untrained', seed, steps: 0, learningRate, trainingLoss: initialTraining, validationLoss: initialValidation,
  task: 'Continue red → green → blue. BOS starts with red; EOS is reserved, never trained. Training prefix lengths 4/7/10, validation lengths 5/8; all 3 cycle phases. This is a finite synthetic pattern task, not a general language model.',
  architecture: '2 pre-norm decoder blocks; width 8; 2 heads of width 4; ReLU feedforward width 16; learned positions (12); vocabulary 5; untied output projection; no dropout.',
  history: [{ step: 0, trainingLoss: initialTraining, validationLoss: initialValidation }],
}
const untrained = exportCheckpoint(model, metadata)
for (let step = 1; step <= steps; step++) {
  const example = examples[Math.floor(random() * examples.length)]
  trainDecoderStep(model, example.input, example.targets, state, learningRate)
  if (step % 30 === 0 || step === steps) {
    const trainingLoss = decoderLoss(model, 'training'), validationLoss = decoderLoss(model, 'validation')
    metadata.history.push({ step, trainingLoss, validationLoss })
    console.log(JSON.stringify({ step, trainingLoss, validationLoss }))
  }
}
const final = metadata.history.at(-1)!
const trained = exportCheckpoint(model, { ...metadata, name: 'Trained (360 steps)', steps, trainingLoss: final.trainingLoss, validationLoss: final.validationLoss })
mkdirSync(new URL('../public/checkpoints/', import.meta.url), { recursive: true })
writeFileSync(new URL('../src/learning/decoder-checkpoints.json', import.meta.url), JSON.stringify({ untrained, trained }, null, 2) + '\n')
writeFileSync(new URL('../public/checkpoints/mini-decoder.json', import.meta.url), JSON.stringify(trained, null, 2) + '\n')
