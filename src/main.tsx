import { StrictMode, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import LearningStudio from './learning/LearningStudio.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Suspense fallback={<div className="studio-loading" role="status">Opening the local model…</div>}><LearningStudio /></Suspense>
  </StrictMode>,
)
