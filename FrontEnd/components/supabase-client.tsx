'use client'

import { createClient } from '@supabase/supabase-js'
import { createContext, useContext, useState, useEffect } from 'react'

// Types
type SupabaseClient = ReturnType<typeof createClient>
type SupabaseContext = {
  supabase: SupabaseClient | null
  isLoaded: boolean
}

// Create a context for the Supabase client
const Context = createContext<SupabaseContext>({ supabase: null, isLoaded: false })

export default function SupabaseProvider({
  children,
}: {
  children: React.ReactNode
}) {
  const [supabase, setSupabase] = useState<SupabaseClient | null>(null)
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    // Ensure we only initialize client-side
    if (typeof window !== 'undefined') {
      const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
      const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
      
      if (supabaseUrl && supabaseKey) {
        const client = createClient(supabaseUrl, supabaseKey, {
          auth: {
            persistSession: true
          }
        })
        
        console.log('Supabase client initialized with URL:', supabaseUrl.substring(0, 20) + '...')
        setSupabase(client)
      } else {
        console.error('Missing Supabase credentials:', {
          url: supabaseUrl ? '✓' : '✗',
          key: supabaseKey ? '✓' : '✗'
        })
      }
      
      setIsLoaded(true)
    }
  }, [])

  return (
    <Context.Provider value={{ supabase, isLoaded }}>
      {children}
    </Context.Provider>
  )
}

export const useSupabase = () => {
  const context = useContext(Context)
  if (context === undefined) {
    throw new Error('useSupabase must be used within a SupabaseProvider')
  }
  return context
}
