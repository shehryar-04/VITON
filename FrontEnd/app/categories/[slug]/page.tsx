import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { getCategoryBySlug, getProductsByCategory } from "@/lib/static-data"
import { notFound } from "next/navigation"
import { ArrowLeft, Eye } from "lucide-react"

export default function CategoryPage({ params }: { params: { slug: string } }) {
  const { slug } = params
  const category = getCategoryBySlug(slug)

  if (!category) {
    notFound()
  }

  const products = getProductsByCategory(slug)

  return (
    <div className="container py-10">
      <div className="mb-8">
        <Link href="/categories" className="text-sm text-muted-foreground hover:underline inline-flex items-center">
          <ArrowLeft className="mr-1 h-4 w-4" /> Back to Products
        </Link>
        <h1 className="text-3xl font-bold mt-2">{category.name}</h1>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {products.map((product) => (
          <div
            key={product.id}
            className="group overflow-hidden rounded-xl border shadow-sm transition-all hover:shadow-md"
          >
            <div className="aspect-[3/4] w-full overflow-hidden relative">
              <Image
                src={product.image || "/placeholder.svg?height=400&width=300&text=Product"}
                alt={product.name}
                fill
                className="object-cover transition-transform duration-300 group-hover:scale-105"
                unoptimized
              />
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-all flex items-center justify-center opacity-0 group-hover:opacity-100">
                <Button asChild variant="secondary" size="sm" className="mr-2">
                  <Link href={`/products/${product.slug}`}>View Details</Link>
                </Button>
                <Button asChild size="sm">
                  <Link href={`/try-on/${product.slug}`}>
                    <Eye className="h-4 w-4 mr-1" /> Try On
                  </Link>
                </Button>
              </div>
            </div>
            <div className="p-4">
              <h3 className="text-lg font-medium">{product.name}</h3>
              <p className="mt-1 text-sm text-muted-foreground">${product.price.toFixed(2)}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
