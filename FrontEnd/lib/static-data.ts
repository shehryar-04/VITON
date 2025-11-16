// Simple file to provide server-side access to the static data
import categories from "@/data/categories.json"
import products from "@/data/products.json"

export type Category = {
  id: number
  name: string
  slug: string
  image: string
}

export type Product = {
  id: number
  name: string
  slug: string
  price: number
  description: string
  image: string
  categorySlug: string
  tryOnImage: string
}

export function getAllCategories(): Category[] {
  return categories
}

export function getCategoryBySlug(slug: string): Category | undefined {
  return categories.find((category) => category.slug === slug)
}

export function getAllProducts(): Product[] {
  return products
}

export function getProductsByCategory(categorySlug: string): Product[] {
  return products.filter((product) => product.categorySlug === categorySlug)
}

export function getProductBySlug(slug: string): Product | undefined {
  return products.find((product) => product.slug === slug)
}
