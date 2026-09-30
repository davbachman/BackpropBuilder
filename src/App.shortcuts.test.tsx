import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { expect, it } from 'vitest'
import App from './App'
import { createModelPreset } from './domain/modelPresets'

it.each(['Enter', 'click'])('creates an editable addition block from the canvas shortcut using %s', async method => {
  const { container } = render(<App initialGraph={createModelPreset('blank')} />)
  fireEvent.doubleClick(container.querySelector('.react-flow__pane')!, { clientX: 300, clientY: 200 })
  const search = within(screen.getByRole('dialog', { name: 'Add a block' })).getByRole('searchbox', { name: 'Search blocks' })
  fireEvent.change(search, { target: { value: '+' } })
  if (method === 'Enter') fireEvent.keyDown(search, { key: 'Enter' })
  else fireEvent.click(screen.getByRole('option', { name: /Arithmetic/ }))
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Arithmetic expression' })).toHaveValue('x1 + x2'))
  expect(container.querySelectorAll('.builder-node .node-handle.target')).toHaveLength(2)
  expect(screen.queryByRole('dialog', { name: 'Add a block' })).not.toBeInTheDocument()
})
