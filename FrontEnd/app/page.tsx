import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { getAllCategories } from "@/lib/static-data"
import { ArrowRight, Shirt } from "lucide-react"

export default function Home() {
  const categories = getAllCategories()

  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <section className="w-full py-12 md:py-24 lg:py-32 bg-background relative">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-secondary/5 dark:from-primary/[0.03] dark:to-secondary/[0.03]"></div>
        <div className="container relative px-4 md:px-6">
          <div className="grid gap-6 lg:grid-cols-2 lg:gap-12 items-center">
            <div className="flex flex-col justify-center space-y-4">
              <div className="space-y-2">
                <h1 className="text-3xl font-bold tracking-tighter sm:text-5xl xl:text-6xl/none bg-clip-text text-transparent bg-gradient-to-r from-foreground to-foreground/70">
                  Virtual Try-On Experience
                </h1>
                <p className="max-w-[600px] text-muted-foreground md:text-xl">
                  Try on clothes virtually before you buy. Upload your photo and see how different styles look on you.
                </p>
              </div>
              <div className="flex flex-col gap-2 min-[400px]:flex-row">
                <Button asChild size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg">
                  <Link href="/categories">
                    Start Trying On
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
                <Button variant="outline" size="lg" asChild className="border-primary/20 hover:bg-primary/10">
                  <Link href="#how-it-works">How It Works</Link>
                </Button>
              </div>
            </div>
            <div className="mx-auto lg:ml-auto flex w-full max-w-[400px] items-center justify-center">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-tr from-primary/10 to-secondary/10 rounded-lg"></div>
                <Image
                  src="/categories/hero.jpeg"
                  alt="Virtual Try-On Demo"
                  width={400}
                  height={400}
                  className="rounded-lg object-cover shadow-lg border dark:border-border relative z-10"
                />
                <div className="absolute -bottom-4 -right-4 bg-background/95 dark:bg-background/95 p-3 rounded-lg shadow-lg dark:border dark:border-border backdrop-blur-sm z-20">
                  <Shirt className="h-8 w-8 text-primary" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Categories Preview */}
      <section className="w-full py-12 md:py-24 lg:py-32 bg-background">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="space-y-2">
              <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Browse Categories</h2>
              <p className="max-w-[900px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                Explore our collection of virtual try-on ready clothing
              </p>
            </div>
          </div>
          <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 py-12 sm:grid-cols-2 md:grid-cols-4">
            {categories.map((category) => (
              <Link
                key={category.id}
                href={`/categories/${category.slug}`}
                className="group relative overflow-hidden rounded-xl shadow-sm hover:shadow-md transition-all duration-300 border bg-card"
              >
                <div className="aspect-square w-full overflow-hidden">
                  <Image
                    src={category.image || "/placeholder.svg"}
                    alt={category.name}
                    width={350}
                    height={350}
                    className="object-cover transition-transform duration-300 group-hover:scale-105"
                  />
                </div>
                <div className="absolute inset-0 flex items-end bg-gradient-to-t from-black/60 to-transparent p-4 text-white">
                  <h3 className="text-lg font-medium">{category.name}</h3>
                </div>
              </Link>
            ))}
          </div>
          <div className="flex justify-center">
            <Button asChild size="lg" className="gap-2">
              <Link href="/categories">
                View All Products
                <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="w-full py-12 md:py-24 lg:py-32 bg-secondary/10 dark:bg-secondary/5">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="space-y-2">
              <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">How It Works</h2>
              <p className="max-w-[900px] text-muted-foreground md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed">
                Try on clothes virtually in three simple steps
              </p>
            </div>
          </div>
          <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 py-12 md:grid-cols-3">
            <div className="flex flex-col items-center space-y-2 rounded-lg border bg-card p-6 shadow-sm transition-colors">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-primary-foreground">
                1
              </div>
              <h3 className="text-xl font-bold">Choose a Product</h3>
              <p className="text-muted-foreground text-center">
                Browse our collection and select the item you want to try on
              </p>
            </div>
            <div className="flex flex-col items-center space-y-2 rounded-lg border bg-card p-6 shadow-sm transition-colors">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-primary-foreground">
                2
              </div>
              <h3 className="text-xl font-bold">Upload Your Photo</h3>
              <p className="text-muted-foreground text-center">
                Upload a front-facing photo of yourself for the virtual try-on
              </p>
            </div>
            <div className="flex flex-col items-center space-y-2 rounded-lg border bg-card p-6 shadow-sm transition-colors">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-primary-foreground">
                3
              </div>
              <h3 className="text-xl font-bold">See the Result</h3>
              <p className="text-muted-foreground text-center">View yourself wearing the selected item virtually</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
