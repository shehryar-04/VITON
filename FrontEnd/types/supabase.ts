export type Json = string | number | boolean | null | { [key: string]: Json | undefined } | Json[]

export interface Database {
  public: {
    Tables: {
      categories: {
        Row: {
          id: number
          name: string
          slug: string
          image_url: string
          created_at: string
        }
        Insert: {
          id?: number
          name: string
          slug: string
          image_url: string
          created_at?: string
        }
        Update: {
          id?: number
          name?: string
          slug?: string
          image_url?: string
          created_at?: string
        }
      }
      products: {
        Row: {
          id: number
          name: string
          slug: string
          price: number
          description: string
          image_url: string
          category_slug: string
          created_at: string
        }
        Insert: {
          id?: number
          name: string
          slug: string
          price: number
          description: string
          image_url: string
          category_slug: string
          created_at?: string
        }
        Update: {
          id?: number
          name?: string
          slug?: string
          price?: number
          description?: string
          image_url?: string
          category_slug?: string
          created_at?: string
        }
      }
      profiles: {
        Row: {
          id: string
          updated_at: string
          username: string | null
          full_name: string | null
          avatar_url: string | null
        }
        Insert: {
          id: string
          updated_at?: string
          username?: string | null
          full_name?: string | null
          avatar_url?: string | null
        }
        Update: {
          id?: string
          updated_at?: string
          username?: string | null
          full_name?: string | null
          avatar_url?: string | null
        }
      }
      try_on_history: {
        Row: {
          id: number
          user_id: string
          product_id: number
          product_name: string
          product_image: string
          created_at: string
        }
        Insert: {
          id?: number
          user_id: string
          product_id: number
          product_name: string
          product_image: string
          created_at?: string
        }
        Update: {
          id?: number
          user_id?: string
          product_id?: number
          product_name?: string
          product_image?: string
          created_at?: string
        }
      }
    }
  }
}
