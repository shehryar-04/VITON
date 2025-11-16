import type React from "react"
import { Lato } from "next/font/google"
import { ThemeProvider } from "@/components/theme-provider"
import { CartProvider } from "@/components/cart-provider"
import { Toaster } from "@/components/ui/toaster"
import Header from "@/components/header"
import Footer from "@/components/footer"
import SupabaseProvider from "@/components/supabase-client"

import "./globals.css"

const lato = Lato({
  weight: ["300", "400", "700", "900"],
  subsets: ["latin"],
})

export const metadata = {
  title: "StyleTry - Virtual Try-On",
  description: "Try on clothes virtually before you buy them",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={lato.className}>
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange>
          <SupabaseProvider>
            <CartProvider>
              <div className="flex min-h-screen flex-col">
                <Header />
                <main className="flex-1">{children}</main>
                <Footer />
              </div>
              <Toaster />
            </CartProvider>
          </SupabaseProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
