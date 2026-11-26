import { activeFilterCount, type Facets, type FacetValue, type FilterKey, type Filters } from '../api'
import { FILTER_TITLES, filterValueLabel } from '../labels'

interface Props {
  facets: Facets | null
  filters: Filters
  onChange: (key: FilterKey, value: string | null) => void
  onReset: () => void
}

const SELECT_KEYS: FilterKey[] = ['subCategory', 'articleType', 'gender', 'colour']

/**
 * Catégorie principale en puces, autres dimensions en listes déroulantes. Les nombres affichés
 * tiennent compte des autres filtres déjà choisis (recherche à facettes).
 */
export function FilterPanel({ facets, filters, onChange, onReset }: Props) {
  if (!facets) return null
  const active = activeFilterCount(filters)

  return (
    <div className="filters">
      <fieldset className="filter">
        <legend className="filter__legend">{FILTER_TITLES.category}</legend>
        <div className="filter__chips">
          <button type="button" className="chip" aria-pressed={filters.category === null} onClick={() => onChange('category', null)}>
            Toutes
          </button>
          {facets.category.map((c) => (
            <button
              key={c.name}
              type="button"
              className="chip"
              aria-pressed={filters.category === c.name}
              onClick={() => onChange('category', c.name)}
            >
              {filterValueLabel('category', c.name)}
              <span className="chip__count">{c.count.toLocaleString('fr-FR')}</span>
            </button>
          ))}
        </div>
      </fieldset>

      {SELECT_KEYS.map((key) => (
        <FacetSelect key={key} facet={key} values={facets[key]} value={filters[key]} onChange={onChange} />
      ))}

      {active > 0 && (
        <button type="button" className="button button--ghost" onClick={onReset}>
          Réinitialiser les filtres ({active})
        </button>
      )}
    </div>
  )
}

interface SelectProps {
  facet: FilterKey
  values: FacetValue[]
  value: string | null
  onChange: (key: FilterKey, value: string | null) => void
}

function FacetSelect({ facet, values, value, onChange }: SelectProps) {
  // La valeur choisie reste proposée même si plus aucun produit ne lui correspond.
  const options = value && !values.some((v) => v.name === value) ? [{ name: value, count: 0 }, ...values] : values
  if (options.length === 0) return null
  return (
    <label className="field">
      <span>{FILTER_TITLES[facet]}</span>
      <select value={value ?? ''} onChange={(e) => onChange(facet, e.target.value || null)}>
        <option value="">Tous</option>
        {options.map((o) => (
          <option key={o.name} value={o.name}>
            {filterValueLabel(facet, o.name)} ({o.count.toLocaleString('fr-FR')})
          </option>
        ))}
      </select>
    </label>
  )
}
