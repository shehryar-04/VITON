'use client'

import { createContext, useContext, useEffect, useState } from 'react'
import { toast } from 'sonner'

export type CartItem = {
  id: string
  name: string
  price: number
  image: string
  quantity: number
}

type CartContextType = {
  items: CartItem[]
  addItem: (item: Omit<CartItem, 'quantity'>) => void
  removeItem: (id: string) => void
  updateQuantity: (id: string, quantity: number) => void
  clearCart: () => void
  total: number
}

const CartContext = createContext<CartContextType | null>(null)

export function CartProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<CartItem[]>([])
  const [isClient, setIsClient] = useState(false)

  // Set isClient to true once component mounts
  useEffect(() => {
    setIsClient(true)
  }, [])

  // Load cart from localStorage on mount, but only after confirming we're on the client
  useEffect(() => {
    if (isClient) {
      const savedCart = localStorage.getItem('cart')
      if (savedCart) {
        try {
          setItems(JSON.parse(savedCart))
        } catch (e) {
          console.error('Failed to parse cart data:', e)
          localStorage.removeItem('cart')
        }
      }
    }
  }, [isClient])
  // Save cart to localStorage whenever it changes, but only on the client
  useEffect(() => {
    if (isClient) {
      localStorage.setItem('cart', JSON.stringify(items))
    }
  }, [items, isClient])

  const addItem = (newItem: Omit<CartItem, 'quantity'>) => {
    setItems((currentItems) => {
      const existingItem = currentItems.find((item) => item.id === newItem.id)
      
      if (existingItem) {
        toast.info('Item already in cart')
        return currentItems.map((item) =>
          item.id === newItem.id
            ? { ...item, quantity: item.quantity + 1 }
            : item
        )
      }

      toast.success('Added to cart')
      return [...currentItems, { ...newItem, quantity: 1 }]
    })
  }

  const removeItem = (id: string) => {
    setItems((currentItems) => currentItems.filter((item) => item.id !== id))
    toast.success('Removed from cart')
  }

  const updateQuantity = (id: string, quantity: number) => {
    if (quantity < 1) return
    setItems((currentItems) =>
      currentItems.map((item) =>
        item.id === id ? { ...item, quantity } : item
      )
    )
  }

  const clearCart = () => {
    setItems([])
    toast.success('Cart cleared')
  }

  const total = items.reduce((acc, item) => acc + item.price * item.quantity, 0)

  return (
    <CartContext.Provider
      value={{ items, addItem, removeItem, updateQuantity, clearCart, total }}
    >
      {children}
    </CartContext.Provider>
  )
}

export function useCart() {
  const context = useContext(CartContext)
  if (!context) {
    throw new Error('useCart must be used within a CartProvider')
  }
  return context
}
