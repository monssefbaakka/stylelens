import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  getFacets, NO_FILTERS, searchByImage, searchSimilar, validateImage,
  type Facets, type FilterKey, type Filters, type SearchItem, type SearchResponse,
} from './api'
import { FilterPanel } from './components/FilterPanel'
import { ResultsGrid } from './components/ResultsGrid'
import { UploadZone } from './components/UploadZone'
import { updateFilters } from './filters'
import { filtersSummary, formatMs } from './labels'

/** Requête en cours : une photo envoyée, ou un produit du catalogue (« articles similaires »). */
type Query =
  | { kind: 'image'; file: File; previewUrl: string }
  | { kind: 'product'; product: SearchItem }

/** Une recherche = requête + filtres ; le résultat affiché doit correspondre à la recherche courante. */
interface Search {
  query: Query
  filters: Filters
  k: number
}

interface Outcome {
  search: Search
  response?: SearchResponse
  error?: string
}

const K_OPTIONS = [12, 24, 48]

export default function App() {
  const [query, setQuery] = useState<Query | null>(null)
  const [filters, setFilters] = useState<Filters>(NO_FILTERS)
  const [k, setK] = useState(K_OPTIONS[0])
  const [facets, setFacets] = useState<Facets | null>(null)
  const [outcome, setOutcome] = useState<Outcome | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)

  // Valeurs proposées par chaque filtre, recalculées selon les filtres déjà choisis.
  useEffect(() => {
    const controller = new AbortController()
    getFacets(filters, controller.signal)
      .then(setFacets)
      .catch(() => {
        // garde les facettes précédentes ; l'erreur éventuelle s'affiche avec la recherche
      })
    return () => controller.abort()
  }, [filters])

  const search = useMemo<Search | null>(() => (query ? { query, filters, k } : null), [query, filters, k])

  // Nouvelle recherche à chaque changement de requête, de filtres ou de nombre de résultats ;
  // la précédente est annulée pour ne jamais afficher une réponse périmée.
  useEffect(() => {
    if (!search) return
    const controller = new AbortController()
    const options = { filters: search.filters, k: search.k }
    const pending = search.query.kind === 'image'
      ? searchByImage(search.query.file, options, controller.signal)
      : searchSimilar(search.query.product.id, options, controller.signal)
    pending
      .then((response) => setOutcome({ search, response }))
      .catch((e: unknown) => {
        if (controller.signal.aborted) return
        setOutcome({ search, error: e instanceof Error ? e.message : 'Erreur inattendue' })
      })
    return () => controller.abort()
  }, [search])

  // Libère l'aperçu de l'image précédente.
  useEffect(() => {
    return () => {
      if (query?.kind === 'image') URL.revokeObjectURL(query.previewUrl)
    }
  }, [query])

  const onFile = useCallback((file: File) => {
    const invalid = validateImage(file)
    setUploadError(invalid)
    if (!invalid) setQuery({ kind: 'image', file, previewUrl: URL.createObjectURL(file) })
  }, [])

  const onSimilar = useCallback((product: SearchItem) => {
    setUploadError(null)
    setQuery({ kind: 'product', product })
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }, [])

  const onFilter = useCallback((key: FilterKey, value: string | null) => {
    setFilters((current) => updateFilters(current, key, value))
  }, [])

  const current = outcome?.search === search ? outcome : null
  const loading = search !== null && current === null
  const response = current?.response ?? null
  const error = uploadError ?? current?.error ?? null
  const results = response?.results ?? []
  const summary = filtersSummary(filters)

  return (
    <div className="app">
      <header className="header">
        <a className="brand" href="/" aria-label="StyleLens, accueil">
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <circle cx="14" cy="14" r="7" fill="none" stroke="currentColor" strokeWidth="3" />
            <path d="M19.5 19.5 25 25" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" />
          </svg>
          StyleLens
        </a>
        <p className="header__tagline">On ne sait pas toujours nommer ce que l'on cherche, mais on sait le montrer.</p>
      </header>

      {!query ? (
        <main className="landing">
          <UploadZone onFile={onFile} />
          {error && <p className="alert" role="alert">{error}</p>}
        </main>
      ) : (
        <main className="layout">
          <aside className="panel" aria-label="Votre recherche">
            <figure className="query">
              <img
                src={query.kind === 'image' ? query.previewUrl : query.product.imageUrl}
                alt={query.kind === 'image' ? 'Votre image' : query.product.name}
              />
              <figcaption>
                {query.kind === 'image' ? 'Votre image' : <>Similaires à <strong>{query.product.name}</strong></>}
              </figcaption>
            </figure>
            <UploadZone onFile={onFile} compact />
            <FilterPanel facets={facets} filters={filters} onChange={onFilter} onReset={() => setFilters(NO_FILTERS)} />
            <label className="field">
              <span>Résultats</span>
              <select value={k} onChange={(e) => setK(Number(e.target.value))}>
                {K_OPTIONS.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
          </aside>

          <section className="results" aria-live="polite">
            <div className="results__head">
              <h2>{summary ?? 'Tous les articles'}</h2>
              {response && (
                <p className="results__meta">
                  {results.length} résultat{results.length > 1 ? 's' : ''} · {formatMs(response.timings.totalMs)}
                </p>
              )}
            </div>
            {error && <p className="alert" role="alert">{error}</p>}
            {response && results.length === 0 && (
              <p className="empty">Aucun article ne correspond{summary ? ` à « ${summary} »` : ''}. Essayez d'enlever un filtre.</p>
            )}
            <ResultsGrid items={results} loading={loading} placeholders={k} onSimilar={onSimilar} />
          </section>
        </main>
      )}
    </div>
  )
}
