import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { getProductBySlug, getCategoryBySlug } from "@/lib/static-data"
import { notFound } from "next/navigation"
import { ArrowLeft, Eye } from "lucide-react"
import { AddToCart } from "@/components/add-to-cart"

export default function ProductPage({ params }: { params: { slug: string } }) {
  const { slug } = params
  const product = getProductBySlug(slug)

  if (!product) {
    notFound()
  }

  const category = getCategoryBySlug(product.categorySlug)

  return (
    <div className="container py-10">
      <div className="mb-8">
        <Link
          href={`/categories/${product.categorySlug}`}
          className="text-sm text-muted-foreground hover:underline inline-flex items-center"
        >
          <ArrowLeft className="mr-1 h-4 w-4" /> Back to {category?.name || "Category"}
        </Link>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        <div className="overflow-hidden rounded-xl border shadow-sm">
          <div className="relative aspect-[3/4] w-full">
            <Image
              src={product.image || "/placeholder.svg?height=600&width=500&text=Product+Image"}
              alt={product.name}
              fill
              className="object-cover"
              unoptimized
            />
          </div>
        </div>

        <div className="flex flex-col">
          <h1 className="text-3xl font-bold">{product.name}</h1>
          <p className="text-2xl font-semibold mt-2 text-primary">${product.price.toFixed(2)}</p>

          <div className="mt-6">
            <h3 className="text-lg font-medium mb-2">Description</h3>
            <p className="text-muted-foreground">{product.description}</p>
          </div>

          <div className="mt-8 flex gap-4">
            <AddToCart product={product} />
            <Button asChild variant="outline" size="lg" className="gap-2">
              <Link href={`/try-on/${product.slug}`}>
                <Eye className="h-5 w-5" />
                Try On
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
