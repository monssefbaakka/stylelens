// Client de catalog-api. En développement, Vite relaie /api et /images vers l'API (vite.config.ts).

export interface Product {
  id: number
  name: string
  gender: string | null
  masterCategory: string
  subCategory: string | null
  articleType: string
  baseColour: string | null
  imageUrl: string
}

export interface SearchItem extends Product {
  /** Similarité cosinus avec la requête (1 = identique). */
  score: number
}

export interface SearchResponse {
  results: SearchItem[]
  query: { category: string | null; k: number; productId: number | null }
  timings: { embeddingMs: number | null; searchMs: number; totalMs: number }
}

export interface Category {
  name: string
  count: number
}

export interface SearchOptions {
  category?: string | null
  k?: number
}

export const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp']
export const MAX_IMAGE_BYTES = 10 * 1024 * 1024

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

const BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? ''

/** Vérification locale avant envoi (mêmes règles que l'API). Renvoie un message d'erreur ou null. */
export function validateImage(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) return 'Format non accepté : utilisez une image JPEG, PNG ou WebP.'
  if (file.size > MAX_IMAGE_BYTES) return 'Image trop lourde (10 Mo maximum).'
  if (file.size === 0) return 'Le fichier est vide.'
  return null
}

function queryString(options: SearchOptions): string {
  const params = new URLSearchParams()
  if (options.category) params.set('category', options.category)
  if (options.k) params.set('k', String(options.k))
  const qs = params.toString()
  return qs ? `?${qs}` : ''
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(BASE_URL + path, init)
  } catch (error) {
    if (init?.signal?.aborted) throw error
    throw new ApiError("Impossible de joindre le serveur. Vérifiez que l'API est démarrée.", 0)
  }
  if (!response.ok) {
    let detail: string | undefined
    try {
      const body = await response.json()
      detail = body.detail || body.title
    } catch {
      // corps absent ou non JSON
    }
    throw new ApiError(detail || `Erreur du serveur (${response.status})`, response.status)
  }
  return response.json() as Promise<T>
}

export function searchByImage(file: File, options: SearchOptions = {}, signal?: AbortSignal) {
  const form = new FormData()
  form.append('image', file)
  return request<SearchResponse>(`/api/search${queryString(options)}`, { method: 'POST', body: form, signal })
}

export function searchSimilar(productId: number, options: SearchOptions = {}, signal?: AbortSignal) {
  return request<SearchResponse>(`/api/products/${productId}/similar${queryString(options)}`, { signal })
}

export function getCategories(signal?: AbortSignal) {
  return request<Category[]>('/api/categories', { signal })
}
