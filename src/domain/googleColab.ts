/** Browser-only Google Drive upload for notebooks the user chooses to open in Colab. */

export const DRIVE_FILE_SCOPE = 'https://www.googleapis.com/auth/drive.file'
export const COLAB_NOTEBOOK_NAME = 'backprop-builder-model.ipynb'
export const COLAB_CLIENT_ID_STORAGE_KEY = 'backprop-builder-google-oauth-client-id'

interface TokenResponse {
  access_token?: string
  expires_in?: number
  scope?: string
  error?: string
  error_description?: string
}

interface TokenClient {
  requestAccessToken(): void
}

interface GoogleIdentity {
  accounts: {
    oauth2: {
      initTokenClient(config: {
        client_id: string
        scope: string
        callback: (response: TokenResponse) => void
        error_callback: (error: { type?: string }) => void
      }): TokenClient
      hasGrantedAllScopes(response: TokenResponse, scope: string): boolean
      revoke(token: string, callback?: () => void): void
    }
  }
}

type GoogleWindow = Window & { google?: GoogleIdentity }

let identityScript: Promise<GoogleIdentity> | undefined

function googleIdentity(): GoogleIdentity | undefined {
  return (window as GoogleWindow).google
}

export function validGoogleClientId(value: string): boolean {
  return /^\d+-[a-zA-Z0-9_-]+\.apps\.googleusercontent\.com$/.test(value.trim())
}

export function configuredGoogleClientId(): string {
  try {
    const saved = window.localStorage.getItem(COLAB_CLIENT_ID_STORAGE_KEY)
    if (saved && validGoogleClientId(saved)) return saved
  } catch {
    // Private browsing can disallow local storage; the field still works for this session.
  }
  const siteClientId = import.meta.env.VITE_GOOGLE_OAUTH_CLIENT_ID?.trim() ?? ''
  return validGoogleClientId(siteClientId) ? siteClientId : ''
}

export function saveGoogleClientId(clientId: string): void {
  try {
    if (clientId) window.localStorage.setItem(COLAB_CLIENT_ID_STORAGE_KEY, clientId)
    else window.localStorage.removeItem(COLAB_CLIENT_ID_STORAGE_KEY)
  } catch {
    // Keeping the configuration in React state is enough for the current session.
  }
}

export function loadGoogleIdentity(): Promise<GoogleIdentity> {
  const ready = googleIdentity()
  if (ready?.accounts?.oauth2) return Promise.resolve(ready)
  if (identityScript) return identityScript
  identityScript = new Promise<GoogleIdentity>((resolve, reject) => {
    const script = document.createElement('script')
    script.src = 'https://accounts.google.com/gsi/client'
    script.async = true
    script.onload = () => {
      const identity = googleIdentity()
      if (identity?.accounts?.oauth2) resolve(identity)
      else reject(new Error('Google sign-in did not initialize. Try another browser or turn off content blocking for this site.'))
    }
    script.onerror = () => reject(new Error('Could not load Google sign-in. Check your connection or content blocker.'))
    document.head.appendChild(script)
  }).catch(error => {
    identityScript = undefined
    throw error
  })
  return identityScript
}

export interface DriveAccess {
  clientId: string
  token: string
  expiresAt: number
}

export function requestDriveAccess(clientId: string): Promise<DriveAccess> {
  if (!validGoogleClientId(clientId)) return Promise.reject(new Error('Enter a valid Google OAuth web client ID.'))
  const identity = googleIdentity()
  if (!identity?.accounts?.oauth2) return Promise.reject(new Error('Google sign-in is still loading. Try Connect again.'))
  return new Promise<DriveAccess>((resolve, reject) => {
    const client = identity.accounts.oauth2.initTokenClient({
      client_id: clientId,
      scope: DRIVE_FILE_SCOPE,
      callback: response => {
        if (response.error) {
          reject(new Error(response.error_description || response.error))
          return
        }
        if (!response.access_token || !identity.accounts.oauth2.hasGrantedAllScopes(response, DRIVE_FILE_SCOPE)) {
          reject(new Error('Google Drive file access was not granted.'))
          return
        }
        resolve({
          clientId,
          token: response.access_token,
          expiresAt: Date.now() + Math.max(0, Number(response.expires_in ?? 3600) - 60) * 1000,
        })
      },
      error_callback: error => reject(new Error(error.type === 'popup_closed' ? 'Google sign-in was closed before connecting.' : 'Google sign-in failed' + (error.type ? ': ' + error.type : '.'))),
    })
    // This runs in the click handler's user gesture; Google opens its own consent dialog.
    client.requestAccessToken()
  })
}

export function revokeDriveAccess(access: DriveAccess): void {
  googleIdentity()?.accounts.oauth2.revoke(access.token)
}

export function activeDriveAccess(access: DriveAccess | undefined, clientId: string): boolean {
  return Boolean(access && access.clientId === clientId && access.expiresAt > Date.now())
}

export async function uploadNotebookToDrive(
  notebook: string,
  access: DriveAccess,
  fetcher: typeof fetch = fetch,
): Promise<{ id: string; colabUrl: string }> {
  if (!activeDriveAccess(access, access.clientId)) throw new Error('Google Drive access expired. Reconnect in File → Colab connection.')
  const boundary = 'backprop-builder-' + Math.random().toString(36).slice(2)
  const body = new Blob([
    '--' + boundary + '\r\n',
    'Content-Type: application/json; charset=UTF-8\r\n\r\n',
    JSON.stringify({ name: COLAB_NOTEBOOK_NAME, mimeType: 'application/x-ipynb+json' }),
    '\r\n--' + boundary + '\r\n',
    'Content-Type: application/x-ipynb+json\r\n\r\n',
    notebook,
    '\r\n--' + boundary + '--\r\n',
  ], { type: 'multipart/related; boundary=' + boundary })
  const response = await fetcher('https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,name', {
    method: 'POST',
    headers: {
      Authorization: 'Bearer ' + access.token,
      'Content-Type': 'multipart/related; boundary=' + boundary,
    },
    body,
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const payload = await response.json() as { error?: { message?: string } }
      detail = payload.error?.message || detail
    } catch {
      // Status is still enough to explain the failure.
    }
    throw new Error('Google Drive upload failed (' + response.status + '): ' + detail)
  }
  const uploaded = await response.json() as { id?: string }
  if (!uploaded.id || !/^[a-zA-Z0-9_-]+$/.test(uploaded.id)) throw new Error('Google Drive did not return a valid notebook ID.')
  return { id: uploaded.id, colabUrl: 'https://colab.research.google.com/drive/' + uploaded.id }
}
