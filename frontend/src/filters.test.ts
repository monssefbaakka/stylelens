import { describe, expect, it } from 'vitest'
import { NO_FILTERS } from './api'
import { updateFilters } from './filters'
import { filtersSummary } from './labels'

describe('updateFilters', () => {
  const selected = { ...NO_FILTERS, category: 'Apparel', subCategory: 'Topwear', articleType: 'Tshirts', gender: 'Women', colour: 'Red' }

  it('clears sub-category and article type when the category changes', () => {
    expect(updateFilters(selected, 'category', 'Footwear')).toEqual({
      ...NO_FILTERS, category: 'Footwear', gender: 'Women', colour: 'Red',
    })
  })

  it('clears the article type when the sub-category changes', () => {
    expect(updateFilters(selected, 'subCategory', 'Bottomwear')).toMatchObject({ category: 'Apparel', subCategory: 'Bottomwear', articleType: null })
  })

  it('keeps the other filters for independent dimensions', () => {
    expect(updateFilters(selected, 'colour', null)).toEqual({ ...selected, colour: null })
    expect(updateFilters(selected, 'category', 'Apparel')).toEqual(selected)
  })
})

describe('filtersSummary', () => {
  it('lists active filters with French labels', () => {
    expect(filtersSummary({ ...NO_FILTERS, category: 'Footwear', gender: 'Men', colour: 'Navy Blue' })).toBe('Chaussures · Homme · Bleu marine')
    expect(filtersSummary({ ...NO_FILTERS, articleType: 'Casual Shoes' })).toBe('Casual Shoes')
    expect(filtersSummary(NO_FILTERS)).toBeNull()
  })
})
