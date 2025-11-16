'use client'

// This file contains client-side only data access methods
import { createClient } from '@supabase/supabase-js'
import { type Category, type Product } from './static-data'

// Helper function to get the Supabase client on the client side
function getSupabaseClient() {
  if (typeof window === 'undefined') {
    console.log('Cannot initialize Supabase on server side')
    return null
  }
  
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  
  if (!supabaseUrl || !supabaseKey) {
    console.error('Missing Supabase environment variables')
    return null
  }
  
  return createClient(supabaseUrl, supabaseKey)
}

export type TryOnHistory = {
  id: string
  user_id?: string
  product_id: number
  product_name: string
  product_image: string
  user_image_url: string
  cloth_image_url: string
  result_image_url: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  error?: string
  created_at: string
  completed_at?: string
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

// Updated to use Supabase
export async function getTryOnHistory(): Promise<TryOnHistory[]> {
  const { data, error } = await supabase
    .from('try_on_history')
    .select('*')
    .order('created_at', { ascending: false })
    .limit(20)

  if (error) {
    console.error('Error fetching try-on history:', error)
    return []
  }

  return data || []
}

export async function saveTryOnHistory(item: Omit<TryOnHistory, "id" | "created_at" | "completed_at">): Promise<void> {
  // Check if URL is available
  if (!supabase) {
    console.error('Supabase client is not initialized properly')
    return
  }

  // Log the data being saved for debugging
  console.log('Saving try-on history:', item)

  try {
    const { error } = await supabase
      .from('try_on_history')
      .insert([{
        ...item,
        created_at: new Date().toISOString(),
      }])

    if (error) {
      console.error('Error saving try-on history:', error)
      throw error
    }
    
    console.log('Try-on history saved successfully')
  } catch (err) {
    console.error('Exception saving try-on history:', err)
    throw err
  }
}

// Add a debugging function to check image paths
export function checkImagePaths(): void {
  // Ensure this only runs in the browser
  if (typeof window === "undefined") return

  console.log("Checking image paths...")
  // Ensure we don't run this during server rendering or hydration
  // Use requestAnimationFrame to ensure DOM is ready
  requestAnimationFrame(() => {
    // Check category images
    categories.forEach((category) => {
      const img = new Image()
      img.onload = () => console.log(`Category image loaded: ${category.image}`)
      img.onerror = () => console.error(`Failed to load category image: ${category.image}`)
      img.src = category.image
    })

    // Check product images
    products.forEach((product) => {
      const img = new Image()
      img.onload = () => console.log(`Product image loaded: ${product.image}`)
      img.onerror = () => console.error(`Failed to load product image: ${product.image}`)
      img.src = product.image

      if (product.tryOnImage) {
        const tryOnImg = new Image()
        tryOnImg.onload = () => console.log(`Try-on image loaded: ${product.tryOnImage}`)
        tryOnImg.onerror = () => console.error(`Failed to load try-on image: ${product.tryOnImage}`)
        tryOnImg.src = product.tryOnImage
      }
    })
  })
}
