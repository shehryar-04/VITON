import { getProductBySlug } from "@/lib/static-data"
import { notFound } from "next/navigation"
import TryOnClient from "@/components/try-on-client"
import { useState } from "react"

export default function TryOnPage({ params }: { params: { slug: string } }) {
  const { slug } = params
  const product = getProductBySlug(slug)

  if (!product) {
    notFound()
  }

  return (
    <div>
      <TryOnClient 
        product={product} 
      />
      <div className="container mt-4 text-sm text-muted-foreground">
        <p>Product Image Path: {product.tryOnImage || product.image}</p>
      </div>
    </div>
  )
}