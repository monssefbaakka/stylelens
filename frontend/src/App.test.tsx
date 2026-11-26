import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import type { Facets, SearchItem, SearchResponse } from './api'
import { NO_FILTERS } from './api'

const sneaker: SearchItem = {
  id: 42, name: 'White Sneaker', gender: 'Men', masterCategory: 'Footwear', subCategory: 'Shoes',
  articleType: 'Casual Shoes', baseColour: 'White', imageUrl: '/images/42.jpg', score: 0.9312,
}
const tee: SearchItem = {
  id: 7, name: 'Red Tee', gender: 'Women', masterCategory: 'Apparel', subCategory: 'Topwear',
  articleType: 'Tshirts', baseColour: 'Red', imageUrl: '/images/7.jpg', score: 0.71,
}

const facets: Facets = {
  category: [{ name: 'Apparel', count: 21000 }, { name: 'Footwear', count: 9000 }],
  subCategory: [{ name: 'Topwear', count: 15000 }, { name: 'Shoes', count: 7000 }],
  articleType: [{ name: 'Tshirts', count: 7000 }, { name: 'Casual Shoes', count: 2800 }],
  gender: [{ name: 'Men', count: 22000 }, { name: 'Women', count: 18000 }],
  colour: [{ name: 'Black', count: 9000 }, { name: 'Red', count: 2400 }],
}

function searchResponse(results: SearchItem[], productId: number | null = null): SearchResponse {
  return { results, query: { filters: NO_FILTERS, k: 12, productId }, timings: { embeddingMs: 30, searchMs: 4, totalMs: 36 } }
}

function json(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }))
}

let fetchMock: ReturnType<typeof vi.fn>
const calls = (prefix: string) => fetchMock.mock.calls.map((c) => String(c[0])).filter((u) => u.startsWith(prefix))

beforeEach(() => {
  fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = String(input)
    if (url.startsWith('/api/filters')) return json(facets)
    if (url.startsWith('/api/products/42/similar')) return json(searchResponse([tee], 42))
    if (url.startsWith('/api/search')) return json(searchResponse([sneaker, tee]))
    return json({ detail: 'inconnu' }, 404)
  })
  vi.spyOn(globalThis, 'fetch').mockImplementation(fetchMock as typeof fetch)
})

const photo = () => new File(['img'], 'photo.jpg', { type: 'image/jpeg' })

async function uploadPhoto(user: ReturnType<typeof userEvent.setup>) {
  render(<App />)
  await user.upload(screen.getByTestId('file-input'), photo())
  return screen.findByRole('list', { name: 'Articles similaires' })
}

describe('App', () => {
  it('searches by uploaded image and shows ranked results with scores', async () => {
    const user = userEvent.setup()
    const grid = await uploadPhoto(user)

    const cards = within(grid).getAllByRole('listitem')
    expect(cards).toHaveLength(2)
    expect(within(cards[0]).getByRole('heading')).toHaveTextContent('White Sneaker')
    expect(within(cards[0]).getByText('93 %')).toBeInTheDocument()
    expect(within(cards[0]).getByRole('img')).toHaveAttribute('src', '/images/42.jpg')
    expect(screen.getByText(/2 résultats · 36 ms/)).toBeInTheDocument()
    expect(screen.getByAltText('Votre image')).toHaveAttribute('src', 'blob:preview')
  })

  it('re-runs the search and the facets when a category is selected', async () => {
    const user = userEvent.setup()
    await uploadPhoto(user)

    await user.click(await screen.findByRole('button', { name: /Chaussures/ }))

    expect(screen.getByRole('button', { name: /Chaussures/ })).toHaveAttribute('aria-pressed', 'true')
    await screen.findByRole('heading', { name: 'Chaussures' })
    expect(calls('/api/search').at(-1)).toBe('/api/search?category=Footwear&k=12')
    expect(calls('/api/filters').at(-1)).toBe('/api/filters?category=Footwear')
  })

  it('combines article type, gender and colour filters', async () => {
    const user = userEvent.setup()
    await uploadPhoto(user)

    await user.selectOptions(await screen.findByLabelText("Type d'article"), 'Casual Shoes')
    await user.selectOptions(screen.getByLabelText('Genre'), 'Men')
    await user.selectOptions(screen.getByLabelText('Couleur'), 'Black')

    await screen.findByRole('heading', { name: 'Casual Shoes · Homme · Noir' })
    expect(calls('/api/search').at(-1)).toBe('/api/search?articleType=Casual+Shoes&gender=Men&colour=Black&k=12')
    expect(screen.getByRole('option', { name: /^Homme \(22\s000\)$/ })).toBeInTheDocument()
  })

  it('resets every filter at once', async () => {
    const user = userEvent.setup()
    await uploadPhoto(user)
    await user.selectOptions(await screen.findByLabelText('Couleur'), 'Red')
    await user.click(screen.getByRole('button', { name: /Chaussures/ }))

    await user.click(screen.getByRole('button', { name: 'Réinitialiser les filtres (2)' }))

    await screen.findByRole('heading', { name: 'Tous les articles' })
    expect(screen.getByLabelText('Couleur')).toHaveValue('')
    expect(calls('/api/search').at(-1)).toBe('/api/search?k=12')
    expect(screen.queryByRole('button', { name: /Réinitialiser/ })).not.toBeInTheDocument()
  })

  it('finds items similar to a result without re-uploading, keeping filters', async () => {
    const user = userEvent.setup()
    const grid = await uploadPhoto(user)
    await user.selectOptions(await screen.findByLabelText('Genre'), 'Women')

    await user.click(within(within(grid).getAllByRole('listitem')[0]).getByRole('button', { name: 'Articles similaires' }))

    expect(await screen.findByText(/Similaires à/)).toHaveTextContent('White Sneaker')
    expect(await screen.findByRole('heading', { name: /Red Tee/ })).toBeInTheDocument()
    expect(calls('/api/products/42/similar')).toContain('/api/products/42/similar?gender=Women&k=12')
  })

  it('rejects unsupported files before calling the API', async () => {
    const user = userEvent.setup({ applyAccept: false })
    render(<App />)
    await user.upload(screen.getByTestId('file-input'), new File(['gif'], 'a.gif', { type: 'image/gif' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/Format non accepté/)
    expect(calls('/api/search')).toHaveLength(0)
  })

  it('shows the API error message', async () => {
    fetchMock.mockImplementation((input: RequestInfo | URL) =>
      String(input).startsWith('/api/filters')
        ? json(facets)
        : json({ title: 'Service IA indisponible', detail: 'Service IA injoignable. Réessayer dans quelques instants.' }, 503))
    const user = userEvent.setup()
    render(<App />)
    await user.upload(screen.getByTestId('file-input'), photo())

    expect(await screen.findByRole('alert')).toHaveTextContent('Service IA injoignable')
  })
})
