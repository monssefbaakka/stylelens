import { describe, expect, it, vi } from 'vitest'
import { ApiError, MAX_IMAGE_BYTES, searchByImage, searchSimilar, validateImage } from './api'

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

describe('validateImage', () => {
  it('accepts JPEG, PNG and WebP', () => {
    for (const type of ['image/jpeg', 'image/png', 'image/webp']) {
      expect(validateImage(new File(['x'], 'a', { type }))).toBeNull()
    }
  })

  it('rejects other types, empty and oversized files', () => {
    expect(validateImage(new File(['x'], 'a.gif', { type: 'image/gif' }))).toMatch(/Format non accepté/)
    expect(validateImage(new File([], 'a.png', { type: 'image/png' }))).toMatch(/vide/)
    const big = new File(['x'], 'big.jpg', { type: 'image/jpeg' })
    Object.defineProperty(big, 'size', { value: MAX_IMAGE_BYTES + 1 })
    expect(validateImage(big)).toMatch(/trop lourde/)
  })
})

describe('searchByImage', () => {
  it('posts the image as multipart with filter parameters', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      jsonResponse({ results: [], query: { category: 'Footwear', k: 24, productId: null }, timings: { totalMs: 5 } }),
    )
    const file = new File(['img'], 'shoe.png', { type: 'image/png' })

    await searchByImage(file, { category: 'Footwear', k: 24 })

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/search?category=Footwear&k=24')
    expect(init?.method).toBe('POST')
    const body = init?.body
    expect(body).toBeInstanceOf(FormData)
    expect((body as FormData).get('image')).toBe(file)
  })

  it('omits empty parameters', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({ results: [] }))
    await searchByImage(new File(['img'], 'a.png', { type: 'image/png' }), { category: null })
    expect(fetchMock.mock.calls[0][0]).toBe('/api/search')
  })

  it('surfaces the problem detail of an API error', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      jsonResponse({ title: 'Service IA indisponible', detail: 'Service IA injoignable.' }, 503),
    )
    const error = await searchByImage(new File(['img'], 'a.png', { type: 'image/png' })).catch((e) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 503, message: 'Service IA injoignable.' })
  })

  it('reports an unreachable server', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(searchSimilar(3)).rejects.toMatchObject({ status: 0, message: /Impossible de joindre/ })
  })

  it('rethrows aborts untouched', async () => {
    const controller = new AbortController()
    controller.abort()
    const abort = new DOMException('Aborted', 'AbortError')
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(abort)
    await expect(searchSimilar(3, {}, controller.signal)).rejects.toBe(abort)
  })
})
