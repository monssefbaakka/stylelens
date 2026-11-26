import type { Category } from '../api'
import { categoryLabel } from '../labels'

interface Props {
  categories: Category[]
  value: string | null
  onChange: (category: string | null) => void
}

export function CategoryFilter({ categories, value, onChange }: Props) {
  if (categories.length === 0) return null
  const options: { name: string | null; label: string; count?: number }[] = [
    { name: null, label: 'Toutes' },
    ...categories.map((c) => ({ name: c.name, label: categoryLabel(c.name), count: c.count })),
  ]
  return (
    <fieldset className="filter">
      <legend className="filter__legend">Catégorie</legend>
      <div className="filter__chips">
        {options.map((o) => (
          <button
            key={o.name ?? 'all'}
            type="button"
            className="chip"
            aria-pressed={value === o.name}
            onClick={() => onChange(o.name)}
          >
            {o.label}
            {o.count !== undefined && <span className="chip__count">{o.count.toLocaleString('fr-FR')}</span>}
          </button>
        ))}
      </div>
    </fieldset>
  )
}
