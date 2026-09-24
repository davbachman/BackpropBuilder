export type LessonKind =
  | 'linear'
  | 'neuron'
  | 'small-network'
  | 'playground'
  | 'probabilities'
  | 'embeddings'
  | 'attention'
  | 'causal'
  | 'block'
  | 'decoder'
  | 'cnn'

export const LESSONS: Array<{
  id: LessonKind
  title: string
  subtitle: string
  question: string
  experiment: string
  level: string
  symbol: string
}> = [
  {
    id: 'linear',
    title: 'A line that learns',
    subtitle: 'Linear prediction & fitting',
    question: 'How does changing the slope move a prediction?',
    experiment:
      'Change only the slope, then compare the same data and training budget.',
    level: '01 · Calculations',
    symbol: '↗',
  },
  {
    id: 'neuron',
    title: 'Inside one neuron',
    subtitle: 'Weights, bias & activation',
    question: 'What happens to a negative number at a ReLU gate?',
    experiment:
      'Switch between identity and ReLU while keeping the weights fixed.',
    level: '02 · Calculations',
    symbol: '◉',
  },
  {
    id: 'small-network',
    title: 'Connect the neurons',
    subtitle: 'A hand-traceable network',
    question: 'How does one weight affect the final prediction?',
    experiment: 'Open a layer, select a neuron, and change one connection.',
    level: '03 · Networks',
    symbol: '⋈',
  },
  {
    id: 'playground',
    title: 'Find a boundary',
    subtitle: 'Network playground',
    question: 'Can a small network separate two diagonal pairs of regions?',
    experiment:
      'Compare hidden layers using the same seed, data, and number of updates.',
    level: '04 · Networks',
    symbol: '⊙',
  },
  {
    id: 'probabilities',
    title: 'Scores become choices',
    subtitle: 'Stable softmax',
    question: 'Does the highest score have to receive all the probability?',
    experiment:
      'Add the same number to every score. Do the probabilities change?',
    level: '05 · Tokens',
    symbol: '▥',
  },
  {
    id: 'embeddings',
    title: 'Give tokens coordinates',
    subtitle: 'Lookup & position information',
    question:
      'Why do two copies of the same token have different representations?',
    experiment:
      'Put the same token in two positions and compare its embedding and position rows.',
    level: '06 · Tokens',
    symbol: '▦',
  },
  {
    id: 'attention',
    title: 'Mix three messages',
    subtitle: 'One-query attention',
    question: 'Can the output change while the attention weights stay fixed?',
    experiment:
      'Change the middle value from [0, 2] to [0, 4], keeping Q and K fixed.',
    level: '07 · Attention',
    symbol: '≋',
  },
  {
    id: 'causal',
    title: 'Look only backward',
    subtitle: 'Causal self-attention',
    question: 'Which positions can the first token attend to?',
    experiment:
      'Change the last key or value. Earlier output rows must stay the same.',
    level: '08 · Attention',
    symbol: '◩',
  },
  {
    id: 'block',
    title: 'Open a transformer block',
    subtitle: 'Attention, residuals & feedforward',
    question: 'What takes the shortcut around each transformation?',
    experiment:
      'Follow a token through attention, then open a feedforward neuron.',
    level: '09 · Transformers',
    symbol: '⊞',
  },
  {
    id: 'decoder',
    title: 'A tiny model, end to end',
    subtitle: 'A complete miniature decoder',
    question: 'How does the model turn a prompt into its next token?',
    experiment:
      'Compare trained and initial parameters, then step through a generation cycle.',
    level: '10 · Transformers',
    symbol: '✳',
  },
  {
    id: 'cnn',
    title: 'Read a handwritten digit',
    subtitle: 'Convolution, filters & feature maps',
    question: 'What does a small filter find when it slides across an image?',
    experiment:
      'Choose a feature-map cell, inspect its input patch, and edit a shared filter weight.',
    level: '11 · Images',
    symbol: '▦',
  },
]

export function isLessonKind(value: unknown): value is LessonKind {
  return LESSONS.some((lesson) => lesson.id === value)
}
