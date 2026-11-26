import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import type { SearchItem, SearchResponse } from './api'

const sneaker: SearchItem = {
  id: 42, name: 'White Sneaker', gender: 'Men', masterCategory: 'Footwear', subCategory: 'Shoes',
  articleType: 'Casual Shoes', baseColour: 'White', imageUrl: '/images/42.jpg', score: 0.9312,
}
const tee: SearchItem = {
  id: 7, name: 'Red Tee', gender: 'Women', masterCategory: 'Apparel', subCategory: 'Topwear',
  articleType: 'Tshirts', baseColour: 'Red', imageUrl: '/images/7.jpg', score: 0.71,
}

function searchResponse(results: SearchItem[], productId: number | null = null): SearchResponse {
  return { results, query: { category: null, k: 12, productId }, timings: { embeddingMs: 30, searchMs: 4, totalMs: 36 } }
}

function json(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }))
}

let fetchMock: ReturnType<typeof vi.fn>

beforeEach(() => {
  fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = String(input)
    if (url === '/api/categories') {
      return json([{ name: 'Apparel', count: 21000 }, { name: 'Footwear', count: 9000 }])
    }
    if (url.startsWith('/api/products/42/similar')) return json(searchResponse([tee], 42))
    if (url.startsWith('/api/search')) return json(searchResponse([sneaker, tee]))
    return json({ detail: 'inconnu' }, 404)
  })
  vi.spyOn(globalThis, 'fetch').mockImplementation(fetchMock as typeof fetch)
})

const photo = () => new File(['img'], 'photo.jpg', { type: 'image/jpeg' })

describe('App', () => {
  it('searches by uploaded image and shows ranked results with scores', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.upload(screen.getByTestId('file-input'), photo())

    const grid = await screen.findByRole('list', { name: 'Articles similaires' })
    const cards = within(grid).getAllByRole('listitem')
    expect(cards).toHaveLength(2)
    expect(within(cards[0]).getByRole('heading')).toHaveTextContent('White Sneaker')
    expect(within(cards[0]).getByText('93 %')).toBeInTheDocument()
    expect(within(cards[0]).getByRole('img')).toHaveAttribute('src', '/images/42.jpg')
    expect(screen.getByText(/2 résultats · 36 ms/)).toBeInTheDocument()
    expect(screen.getByAltText('Votre image')).toHaveAttribute('src', 'blob:preview')
  })

  it('re-runs the search when a category is selected', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.upload(screen.getByTestId('file-input'), photo())
    await screen.findByRole('list', { name: 'Articles similaires' })

    await user.click(screen.getByRole('button', { name: /Chaussures/ }))

    expect(screen.getByRole('button', { name: /Chaussures/ })).toHaveAttribute('aria-pressed', 'true')
    await screen.findByRole('heading', { name: 'Chaussures' })
    const lastSearch = fetchMock.mock.calls.map((c) => String(c[0])).filter((u) => u.startsWith('/api/search')).at(-1)
    expect(lastSearch).toBe('/api/search?category=Footwear&k=12')
  })

  it('finds items similar to a result without re-uploading', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.upload(screen.getByTestId('file-input'), photo())
    const grid = await screen.findByRole('list', { name: 'Articles similaires' })

    await user.click(within(within(grid).getAllByRole('listitem')[0]).getByRole('button', { name: 'Articles similaires' }))

    expect(await screen.findByText(/Similaires à/)).toHaveTextContent('White Sneaker')
    expect(await screen.findByRole('heading', { name: /Red Tee/ })).toBeInTheDocument()
    expect(fetchMock.mock.calls.some((c) => String(c[0]) === '/api/products/42/similar?k=12')).toBe(true)
  })

  it('rejects unsupported files before calling the API', async () => {
    const user = userEvent.setup({ applyAccept: false })
    render(<App />)
    await user.upload(screen.getByTestId('file-input'), new File(['gif'], 'a.gif', { type: 'image/gif' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/Format non accepté/)
    expect(fetchMock.mock.calls.some((c) => String(c[0]).startsWith('/api/search'))).toBe(false)
  })

  it('shows the API error message', async () => {
    fetchMock.mockImplementation((input: RequestInfo | URL) =>
      String(input) === '/api/categories'
        ? json([])
        : json({ title: 'Service IA indisponible', detail: 'Service IA injoignable. Réessayer dans quelques instants.' }, 503))
    const user = userEvent.setup()
    render(<App />)
    await user.upload(screen.getByTestId('file-input'), photo())

    expect(await screen.findByRole('alert')).toHaveTextContent('Service IA injoignable')
  })
})
