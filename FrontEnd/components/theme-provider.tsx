'use client'

import * as React from 'react'
import {
  ThemeProvider as NextThemesProvider,
  type ThemeProviderProps,
} from 'next-themes'

export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  // Use suppressHydrationWarning to prevent hydration warnings related to theme
  return (
    <NextThemesProvider {...props} enableSystem={true} enableColorScheme={true}>
      {children}
    </NextThemesProvider>
  )
}
