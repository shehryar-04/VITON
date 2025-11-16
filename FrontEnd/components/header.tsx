"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Menu, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useState } from "react"
import { ModeToggle } from "./mode-toggle"
import { Cart } from "./cart"

export default function Header() {
  const pathname = usePathname()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 shadow-sm">
      <div className="container flex h-16 items-center">
        <Link href="/" className="flex items-center">
          <span className="text-xl font-bold text-primary">StyleShift</span>
        </Link>

        {/* Mobile menu button */}
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden ml-auto"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        >
          {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          <span className="sr-only">Toggle menu</span>
        </Button>

        {/* Desktop navigation */}
        <nav className="hidden md:flex ml-auto gap-4 sm:gap-6">
          
          <Link
            href="/"
            className={`text-sm font-medium ${
              pathname === "/" ? "text-primary" : "text-foreground/60"
            } transition-colors hover:text-primary`}
          >
            Home
          </Link>
          <Link
            href="/about"
            className={`text-sm font-medium ${
              pathname === "/about" ? "text-primary" : "text-foreground/60"
            } transition-colors hover:text-primary`}
          >
            About
          </Link>
          <Link
            href="/categories"
            className={`text-sm font-medium ${
              pathname === "/categories" || pathname.startsWith("/categories/") ? "text-primary" : "text-foreground/60"
            } transition-colors hover:text-primary`}
          >
            Products
          </Link>
          <Link
            href="/try-history"
            className={`text-sm font-medium ${
              pathname === "/try-history" ? "text-primary" : "text-foreground/60"
            } transition-colors hover:text-primary`}
          >
            Try-On History
          </Link>
          
        </nav>

        <div className="hidden md:flex ml-4 gap-2">
          <ModeToggle />
          <Cart />
        </div>

        {/* Mobile navigation */}
        {mobileMenuOpen && (
          <div className="absolute top-16 left-0 right-0 bg-background border-b p-4 md:hidden shadow-md">
            <nav className="flex flex-col gap-2">
              <Link
                href="/"
                className={`px-2 py-1 rounded-md ${
                  pathname === "/" ? "bg-primary/10 text-primary" : "text-foreground/60"
                } transition-colors hover:text-primary`}
                onClick={() => setMobileMenuOpen(false)}
              >
                Home
              </Link>
              <Link
                href="/categories"
                className={`px-2 py-1 rounded-md ${
                  pathname === "/categories" || pathname.startsWith("/categories/")
                    ? "bg-primary/10 text-primary"
                    : "text-foreground/60"
                } transition-colors hover:text-primary`}
                onClick={() => setMobileMenuOpen(false)}
              >
                Categories
              </Link>
              <Link
                href="/try-history"
                className={`px-2 py-1 rounded-md ${
                  pathname === "/try-history" ? "bg-primary/10 text-primary" : "text-foreground/60"
                } transition-colors hover:text-primary`}
                onClick={() => setMobileMenuOpen(false)}
              >
                Try-On History
              </Link>
              <Link
                href="/about"
                className={`px-2 py-1 rounded-md ${
                  pathname === "/about" ? "bg-primary/10 text-primary" : "text-foreground/60"
                } transition-colors hover:text-primary`}
                onClick={() => setMobileMenuOpen(false)}
              >
                About
              </Link>
              <div className="flex items-center gap-2 mt-2 pt-2 border-t">
                <ModeToggle />
                <Cart />
              </div>
            </nav>
          </div>
        )}
      </div>
    </header>
  )
}
