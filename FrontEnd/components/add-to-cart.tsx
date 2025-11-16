'use client'

import { Button } from "@/components/ui/button"
import { useCart } from "@/components/cart-provider"
import { ShoppingCart } from "lucide-react"

type AddToCartProps = {
  product: {
    id: number
    name: string
    price: number
    image: string
  }
}

export function AddToCart({ product }: AddToCartProps) {
  const { addItem } = useCart()

  return (
    <Button
      onClick={() => addItem({
        id: String(product.id),
        name: product.name,
        price: product.price,
        image: product.image
      })}
      className="w-full gap-2"
      size="lg"
    >
      <ShoppingCart className="h-5 w-5" />
      Add to Cart
    </Button>
  )
}
