import { useState, useEffect } from 'react'

/**
 * Generate a LiveKit access token for the viewer
 * In production, this should come from a backend API
 * For local development, we generate it client-side
 */
export function useToken(roomName: string, identity: string): string | null {
  const [token, setToken] = useState<string | null>(null)

  useEffect(() => {
    // For local development, we'll generate a simple token
    // In production, fetch this from your backend
    generateLocalToken(roomName, identity).then(setToken)
  }, [roomName, identity])

  return token
}

async function generateLocalToken(roomName: string, identity: string): Promise<string> {
  // Simple JWT generation for local development
  // Uses the same credentials as config.yaml
  const apiKey = 'devkey'
  const apiSecret = 'devsecret'

  const header = {
    alg: 'HS256',
    typ: 'JWT',
  }

  const now = Math.floor(Date.now() / 1000)
  const payload = {
    iss: apiKey,
    sub: identity,
    iat: now,
    exp: now + 86400, // 24 hours
    nbf: now,
    video: {
      room: roomName,
      roomJoin: true,
      canSubscribe: true,
      canPublish: false,
    },
  }

  // Base64URL encode
  const base64url = (data: object) => {
    const str = JSON.stringify(data)
    const base64 = btoa(str)
    return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
  }

  const headerB64 = base64url(header)
  const payloadB64 = base64url(payload)

  // For local development with known secret, we can use SubtleCrypto
  const encoder = new TextEncoder()
  const key = await crypto.subtle.importKey(
    'raw',
    encoder.encode(apiSecret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign']
  )

  const signature = await crypto.subtle.sign(
    'HMAC',
    key,
    encoder.encode(`${headerB64}.${payloadB64}`)
  )

  const signatureB64 = btoa(String.fromCharCode(...new Uint8Array(signature)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '')

  return `${headerB64}.${payloadB64}.${signatureB64}`
}
