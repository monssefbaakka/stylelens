import type { FilterKey, Filters } from './api'
import { FILTER_KEYS } from './api'

// Libellés français des valeurs du jeu de données ; les autres valeurs restent en anglais.
const CATEGORY_LABELS: Record<string, string> = {
  Apparel: 'Vêtements',
  Footwear: 'Chaussures',
  Accessories: 'Accessoires',
}

const GENDER_LABELS: Record<string, string> = {
  Men: 'Homme',
  Women: 'Femme',
  Boys: 'Garçon',
  Girls: 'Fille',
  Unisex: 'Mixte',
}

const COLOUR_LABELS: Record<string, string> = {
  Black: 'Noir',
  White: 'Blanc',
  Blue: 'Bleu',
  'Navy Blue': 'Bleu marine',
  Red: 'Rouge',
  Green: 'Vert',
  Grey: 'Gris',
  Brown: 'Marron',
  Pink: 'Rose',
  Purple: 'Violet',
  Yellow: 'Jaune',
  Orange: 'Orange',
  Beige: 'Beige',
  Silver: 'Argenté',
  Gold: 'Doré',
  Maroon: 'Bordeaux',
  Olive: 'Olive',
  Cream: 'Crème',
  Multi: 'Multicolore',
}

export const FILTER_TITLES: Record<FilterKey, string> = {
  category: 'Catégorie',
  subCategory: 'Sous-catégorie',
  articleType: "Type d'article",
  gender: 'Genre',
  colour: 'Couleur',
}

export function categoryLabel(name: string): string {
  return CATEGORY_LABELS[name] ?? name
}

export function filterValueLabel(key: FilterKey, value: string): string {
  if (key === 'category') return categoryLabel(value)
  if (key === 'gender') return GENDER_LABELS[value] ?? value
  if (key === 'colour') return COLOUR_LABELS[value] ?? value
  return value
}

/** Résumé des filtres actifs, ex. « Chaussures · Homme · Noir ». */
export function filtersSummary(filters: Filters): string | null {
  const parts = FILTER_KEYS.flatMap((key) => (filters[key] ? [filterValueLabel(key, filters[key])] : []))
  return parts.length ? parts.join(' · ') : null
}

export function formatScore(score: number): string {
  return `${Math.round(Math.max(0, Math.min(1, score)) * 100)} %`
}

export function formatMs(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}
