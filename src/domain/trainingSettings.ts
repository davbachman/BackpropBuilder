export interface TrainingSettings {
  engine: 'trace' | 'tensor'
  backend: 'auto' | 'cpu' | 'webgl' | 'webgpu'
  optimizer: 'sgd' | 'adam' | 'adamw'
  weightDecay: number
  clipNorm: number
  patience: number
  minDelta: number
}
export const DEFAULT_TRAINING: TrainingSettings = {engine:'trace',backend:'auto',optimizer:'adamw',weightDecay:0.01,clipNorm:1,patience:3,minDelta:0.001}
export function isTrainingSettings(value: unknown): value is TrainingSettings {
  if (!value || typeof value !== 'object') return false
  const v = value as TrainingSettings
  return ['trace','tensor'].includes(v.engine) && ['auto','cpu','webgl','webgpu'].includes(v.backend) && ['sgd','adam','adamw'].includes(v.optimizer) && [v.weightDecay,v.clipNorm,v.minDelta].every(n=>Number.isFinite(n) && n>=0) && Number.isInteger(v.patience) && v.patience>=0
}
