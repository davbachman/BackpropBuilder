import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { activeDriveAccess, COLAB_CLIENT_ID_STORAGE_KEY, configuredGoogleClientId, DRIVE_FILE_SCOPE, requestDriveAccess, saveGoogleClientId, uploadNotebookToDrive, validGoogleClientId } from './googleColab'

const CLIENT_ID = '123456789-demo.apps.googleusercontent.com'

beforeEach(() => {
  const values = new Map<string, string>()
  vi.stubGlobal('localStorage', {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value) },
    removeItem: (key: string) => { values.delete(key) },
  })
})

afterEach(() => vi.unstubAllGlobals())

describe('direct Colab connection', () => {
  it('accepts a public web client ID and saves only that configuration', () => {
    expect(validGoogleClientId(CLIENT_ID)).toBe(true)
    expect(validGoogleClientId('client-secret')).toBe(false)
    saveGoogleClientId(CLIENT_ID)
    expect(configuredGoogleClientId()).toBe(CLIENT_ID)
    expect(window.localStorage.getItem(COLAB_CLIENT_ID_STORAGE_KEY)).toBe(CLIENT_ID)
  })

  it('requests only the Drive file scope from a user-initiated Google token client', async () => {
    const initTokenClient = vi.fn((config: { client_id: string; scope: string; callback: (response: { access_token: string; expires_in: number }) => void }) => ({
      requestAccessToken: () => config.callback({ access_token: 'short-lived-token', expires_in: 3600 }),
    }))
    vi.stubGlobal('google', { accounts: { oauth2: { initTokenClient, hasGrantedAllScopes: () => true } } })
    const access = await requestDriveAccess(CLIENT_ID)
    expect(initTokenClient).toHaveBeenCalledWith(expect.objectContaining({ client_id: CLIENT_ID, scope: DRIVE_FILE_SCOPE }))
    expect(access.token).toBe('short-lived-token')
    expect(activeDriveAccess(access, CLIENT_ID)).toBe(true)
    expect(activeDriveAccess(access, 'another-client')).toBe(false)
  })

  it('uploads notebook bytes as one Drive file and returns its Colab URL', async () => {
    const fetcher = vi.fn(async () => ({ ok: true, json: async () => ({ id: 'drive-file_123' }) }))
    const access = { clientId: CLIENT_ID, token: 'drive-token', expiresAt: Date.now() + 60_000 }
    const result = await uploadNotebookToDrive('{"cells":[]}', access, fetcher as unknown as typeof fetch)
    expect(result.colabUrl).toBe('https://colab.research.google.com/drive/drive-file_123')
    const [url, options] = fetcher.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toContain('uploadType=multipart')
    expect((options.headers as Record<string, string>).Authorization).toBe('Bearer drive-token')
    const body = await (options.body as Blob).text()
    expect(body).toContain('backprop-builder-model.ipynb')
    expect(body).toContain('application/x-ipynb+json')
    expect(body).toContain('{"cells":[]}')
  })

  it('reports Drive errors without claiming the notebook opened', async () => {
    const fetcher = vi.fn(async () => ({ ok: false, status: 403, statusText: 'Forbidden', json: async () => ({ error: { message: 'Drive API is disabled' } }) }))
    const access = { clientId: CLIENT_ID, token: 'drive-token', expiresAt: Date.now() + 60_000 }
    await expect(uploadNotebookToDrive('{}', access, fetcher as unknown as typeof fetch)).rejects.toThrow('Drive API is disabled')
  })
})
