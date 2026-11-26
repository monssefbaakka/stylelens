// Libellés français des catégories du jeu de données (valeurs de masterCategory).
const CATEGORY_LABELS: Record<string, string> = {
  Apparel: 'Vêtements',
  Footwear: 'Chaussures',
  Accessories: 'Accessoires',
}

export function categoryLabel(name: string): string {
  return CATEGORY_LABELS[name] ?? name
}

export function formatScore(score: number): string {
  return `${Math.round(Math.max(0, Math.min(1, score)) * 100)} %`
}

export function formatMs(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}
