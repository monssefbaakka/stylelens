import type { FilterKey, Filters } from './api'

/**
 * Filtres dépendants : changer de catégorie efface la sous-catégorie et le type d'article,
 * changer de sous-catégorie efface le type d'article (sinon la combinaison serait souvent vide).
 */
export function updateFilters(filters: Filters, key: FilterKey, value: string | null): Filters {
  const next = { ...filters, [key]: value }
  if (key === 'category' && value !== filters.category) {
    next.subCategory = null
    next.articleType = null
  }
  if (key === 'subCategory' && value !== filters.subCategory) next.articleType = null
  return next
}
