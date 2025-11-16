import Link from "next/link"
import Image from "next/image"
import { getAllCategories } from "@/lib/static-data"

export default function CategoriesPage() {
  const categories = getAllCategories()

  return (
    <div className="container py-10">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold mb-2">Clothing Products</h1>
        <p className="text-muted-foreground">Select a product to explore our collection</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {categories.map((category) => (
          <Link
            key={category.id}
            href={`/categories/${category.slug}`}
            className="group overflow-hidden rounded-xl border shadow-sm transition-all hover:shadow-md"
          >
            <div className="aspect-[3/4] w-full overflow-hidden relative">
              <Image
                src={category.image || `/placeholder.svg?height=400&width=300&text=${category.name}`}
                alt={category.name}
                fill
                className="object-cover transition-transform group-hover:scale-105"
                unoptimized
              />
            </div>
            <div className="p-4">
              <h3 className="text-lg font-medium">{category.name}</h3>
              <p className="mt-1 text-sm text-muted-foreground">Try on {category.name.toLowerCase()}</p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
