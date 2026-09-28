import { act, fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import { TextImportDialog } from './TextImportDialog'

const reviews = 'text,label,split\ngood,positive,train\nbad,negative,train\nok,positive,test\n'

function deferredFile(name: string) {
  let resolve!: (text: string) => void
  let reject!: (error: Error) => void
  const promise = new Promise<string>((done, failed) => { resolve = done; reject = failed })
  const file = Object.assign(new File([], name), { text: () => promise })
  return { file, promise, resolve, reject }
}

function startImport(file: File) {
  fireEvent.change(screen.getByLabelText('Choose text data file'), { target: { files: [file] } })
  fireEvent.click(screen.getByRole('button', { name: 'Import text dataset' }))
}

it.each(['cancel', 'escape', 'unmount'])('discards a pending file read after %s', async action => {
  const onImport = vi.fn(), onCancel = vi.fn(), pending = deferredFile('reviews.csv')
  const { unmount } = render(<TextImportDialog onImport={onImport} onCancel={onCancel} />)
  startImport(pending.file)
  if (action === 'cancel') fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  else if (action === 'escape') fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
  else unmount()
  await act(async () => { pending.resolve(reviews); await pending.promise })
  expect(onImport).not.toHaveBeenCalled()
  if (action !== 'unmount') expect(onCancel).toHaveBeenCalledOnce()
})

it('keeps the latest file request busy when an older file read fails', async () => {
  const onImport = vi.fn(), first = deferredFile('first.csv'), second = deferredFile('second.csv')
  render(<TextImportDialog onImport={onImport} onCancel={vi.fn()} />)
  startImport(first.file)
  startImport(second.file)
  await act(async () => { first.reject(new Error('Old file failed')); await first.promise.catch(() => undefined) })
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Preparing…' })).toBeDisabled()
  expect(onImport).not.toHaveBeenCalled()
  await act(async () => { second.resolve(reviews); await second.promise })
  expect(onImport).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ fileName: 'second.csv' }))
})
