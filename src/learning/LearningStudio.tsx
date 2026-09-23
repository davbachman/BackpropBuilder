import { lazy, useState } from 'react'
import {
  ArrowLeft,
  ArrowUpRight,
  Braces,
  GitBranch,
  Layers3,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react'
import { createEmptyGraph } from '../domain/examples'
import type { GraphModel } from '../domain/types'
import { LESSONS, type LessonKind } from './presets'
import './studio.css'

const App = lazy(() => import('../App'))
const NetworkLesson = lazy(() =>
  import('./NetworkLesson').then((module) => ({
    default: module.NetworkLesson,
  })),
)
const TransformerLesson = lazy(() =>
  import('./TransformerLesson').then((module) => ({
    default: module.TransformerLesson,
  })),
)

export default function LearningStudio() {
  const [lesson, setLesson] = useState<LessonKind | 'builder' | null>(null)
  const [builderGraph, setBuilderGraph] = useState<GraphModel>(() =>
    createEmptyGraph(),
  )
  const [navigation, setNavigation] = useState(true)
  const openBuilder = (graph: GraphModel) => {
    setBuilderGraph(graph)
    setLesson('builder')
  }
  if (lesson === 'builder')
    return <App initialGraph={builderGraph} onGallery={() => setLesson(null)} />
  const preset = LESSONS.find((item) => item.id === lesson)
  return (
    <div className={`studio ${lesson ? 'studio-active' : 'studio-gallery'}`}>
      <header className="studio-header">
        <button
          className="studio-brand"
          onClick={() => setLesson(null)}
          aria-label="BackpropBuilder preset gallery"
        >
          <span className="studio-logo">
            <GitBranch size={22} />
          </span>
          <span>
            Backprop<span className="brand-light">Builder</span>
          </span>
        </button>
        <span className="studio-header-caption">Small numbers. Big ideas.</span>
        <div className="studio-header-actions">
          {lesson && (
            <button onClick={() => setLesson(null)}>
              <ArrowLeft size={15} /> Presets
            </button>
          )}
          <button onClick={() => openBuilder(createEmptyGraph())}>
            <Braces size={16} /> Blank builder
          </button>
        </div>
      </header>
      {!lesson ? (
        <main className="gallery-main">
          <section className="gallery-hero">
            <div>
              <p className="studio-kicker">
                An open notebook for neural networks
              </p>
              <h1>
                Understand the model.
                <br />
                <span>One calculation at a time.</span>
              </h1>
              <p className="hero-copy">
                Start with a single weight. Follow the numbers through a
                network. Open a transformer and see where its next token comes
                from.
              </p>
              <div className="hero-meta">
                <span>
                  <Layers3 size={16} /> 10 connected explorations
                </span>
                <span className="local-dot">Runs entirely on your device</span>
              </div>
            </div>
            <div className="hero-diagram" aria-hidden="true">
              <svg viewBox="0 0 380 210">
                <g fill="none" stroke="#d8cff0" strokeWidth="2">
                  {[55, 105, 155].flatMap((y, i) =>
                    [40, 85, 130, 175].map((z, j) => (
                      <path
                        key={`${i}-${j}`}
                        d={`M 66 ${y} C 140 ${y}, 140 ${z}, 206 ${z}`}
                      />
                    )),
                  )}
                  {[40, 85, 130, 175].map((y, i) => (
                    <path
                      key={i}
                      d={`M 206 ${y} C 268 ${y}, 268 105, 330 105`}
                    />
                  ))}
                </g>
                {[55, 105, 155].map((y, i) => (
                  <g key={i}>
                    <circle
                      cx="66"
                      cy={y}
                      r="16"
                      fill="#f2eeff"
                      stroke="#ae97de"
                    />
                    <text
                      x="66"
                      y={y + 4}
                      textAnchor="middle"
                      fill="#7151a3"
                      fontSize="12"
                    >
                      {[2, 1, 3][i]}
                    </text>
                  </g>
                ))}
                {[40, 85, 130, 175].map((y, i) => (
                  <circle
                    key={i}
                    cx="206"
                    cy={y}
                    r="16"
                    fill={i === 1 ? '#7950b8' : 'white'}
                    stroke="#aa90d7"
                  />
                ))}
                <circle
                  cx="330"
                  cy="105"
                  r="22"
                  fill="#e5f5ed"
                  stroke="#6ea689"
                />
                <text
                  x="330"
                  y="110"
                  textAnchor="middle"
                  fill="#28754e"
                  fontSize="13"
                >
                  0.8
                </text>
              </svg>
              <div className="hero-equation">
                2 × 0.5 + 1 <span>→</span> a little arithmetic, connected.
              </div>
            </div>
          </section>
          <div className="gallery-section-heading">
            <div>
              <p className="studio-kicker">Choose a starting point</p>
              <h2>Make a prediction. Then open it up.</h2>
            </div>
            <span>No prerequisites beyond curiosity.</span>
          </div>
          <section className="preset-grid" aria-label="Learning presets">
            {LESSONS.map((item) => (
              <button
                className="preset-card"
                key={item.id}
                onClick={() => setLesson(item.id)}
              >
                <div className="preset-top">
                  <span className={`preset-symbol preset-${item.id}`}>
                    {item.symbol}
                  </span>
                  <ArrowUpRight size={17} />
                </div>
                <p className="preset-level">{item.level}</p>
                <h3>{item.title}</h3>
                <p className="preset-subtitle">{item.subtitle}</p>
                <p className="preset-question">{item.question}</p>
              </button>
            ))}
          </section>
          <footer className="gallery-footer">
            <span>
              Created by David Bachman · Built for exploration, discussion, and
              the occasional “aha.”
            </span>
            <a
              href="https://github.com/davbachman/BackpropBuilder"
              target="_blank"
              rel="noreferrer"
            >
              Source & documentation ↗
            </a>
          </footer>
        </main>
      ) : (
        <div
          className={`studio-workbench ${navigation ? '' : 'navigation-hidden'}`}
        >
          {navigation && (
            <aside className="studio-sidebar">
              <p className="studio-kicker">Explorations</p>
              <nav aria-label="Preset navigation">
                {LESSONS.map((item, index) => (
                  <button
                    key={item.id}
                    className={lesson === item.id ? 'selected' : ''}
                    onClick={() => setLesson(item.id)}
                  >
                    <span>{String(index + 1).padStart(2, '0')}</span>
                    {item.subtitle}
                  </button>
                ))}
              </nav>
              <div className="sidebar-note">
                <strong>One model, many views.</strong>
                <p>
                  Opening a component reveals its calculation. It never changes
                  the architecture or its parameters.
                </p>
              </div>
            </aside>
          )}
          <main className="lesson-main">
            <div className="lesson-heading">
              <button
                className="icon-button"
                aria-label={
                  navigation
                    ? 'Hide preset navigation'
                    : 'Show preset navigation'
                }
                onClick={() => setNavigation(!navigation)}
              >
                {navigation ? (
                  <PanelLeftClose size={19} />
                ) : (
                  <PanelLeftOpen size={19} />
                )}
              </button>
              <div>
                <p className="studio-kicker">{preset?.level}</p>
                <h1>{preset?.title}</h1>
              </div>
              <span className="lesson-local">● Local computation</span>
            </div>
            {lesson === 'linear' ||
            lesson === 'neuron' ||
            lesson === 'small-network' ||
            lesson === 'playground' ? (
              <NetworkLesson
                key={lesson}
                kind={lesson}
                onBuilder={openBuilder}
              />
            ) : (
              <TransformerLesson key={lesson} kind={lesson} />
            )}
          </main>
        </div>
      )}
    </div>
  )
}
