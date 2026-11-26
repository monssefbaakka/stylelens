import type { SearchItem } from '../api'
import { categoryLabel, formatScore } from '../labels'

interface Props {
  items: SearchItem[]
  loading: boolean
  placeholders: number
  onSimilar: (item: SearchItem) => void
}

export function ResultsGrid({ items, loading, placeholders, onSimilar }: Props) {
  if (loading) {
    return (
      <ul className="grid" aria-busy="true" aria-label="Chargement des résultats">
        {Array.from({ length: Math.min(placeholders, 12) }, (_, i) => (
          <li key={i} className="card card--skeleton" aria-hidden="true">
            <div className="card__media" />
            <div className="card__body">
              <span className="skeleton-line" />
              <span className="skeleton-line skeleton-line--short" />
            </div>
          </li>
        ))}
      </ul>
    )
  }
  return (
    <ul className="grid" aria-label="Articles similaires">
      {items.map((item, rank) => (
        <li key={item.id} className="card">
          <div className="card__media">
            <img
              src={item.imageUrl}
              alt={item.name}
              loading="lazy"
              onError={(e) => e.currentTarget.classList.add('card__img--missing')}
            />
            <span className="card__score" title="Similarité cosinus avec votre image">
              {formatScore(item.score)}
            </span>
          </div>
          <div className="card__body">
            <h3 className="card__title" title={item.name}>
              <span className="visually-hidden">{rank + 1}. </span>
              {item.name}
            </h3>
            <p className="card__meta">
              {item.articleType}
              {item.baseColour && ` · ${item.baseColour}`}
              {` · ${categoryLabel(item.masterCategory)}`}
            </p>
            <button type="button" className="card__action" onClick={() => onSimilar(item)}>
              Articles similaires
            </button>
          </div>
        </li>
      ))}
    </ul>
  )
}
