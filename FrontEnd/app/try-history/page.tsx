"use client"

import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { useEffect, useState, useCallback } from "react"
import { formatDistanceToNow } from "@/lib/utils"
import { 
  ArrowDownUp, 
  Download, 
  Eye, 
  Filter, 
  History, 
  RefreshCw, 
  Search, 
  Share2,
  Trash2, 
  X 
} from "lucide-react"
import { 
  getHistory, 
  deleteHistoryItem, 
  bulkDeleteHistoryItems, 
  TryOnHistoryItem 
} from "@/lib/supabase-helpers"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogClose, DialogTitle } from "@/components/ui/dialog"
import { useToast } from "@/components/ui/use-toast"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"

export default function TryHistoryPage() {
  const [history, setHistory] = useState<TryOnHistoryItem[]>([])
  const [filteredHistory, setFilteredHistory] = useState<TryOnHistoryItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [previewImage, setPreviewImage] = useState<string | null>(null)
  const [deleteItemId, setDeleteItemId] = useState<string | null>(null)
  const [filterStatus, setFilterStatus] = useState<string>("all")
  const [searchTerm, setSearchTerm] = useState<string>("")
  const [sortOrder, setSortOrder] = useState<"newest" | "oldest">("newest")
  const [currentPage, setCurrentPage] = useState(1)
  const [selectedItems, setSelectedItems] = useState<string[]>([])
  const [isSelectionMode, setIsSelectionMode] = useState(false)
  const [isBulkDeleting, setIsBulkDeleting] = useState(false)
  const itemsPerPage = 5
  const { toast } = useToast()

  // Load history data from Supabase
  const loadHistory = useCallback(async () => {
    setIsLoading(true)
    try {
      const data = await getHistory()
      setHistory(data)
      applyFilters(data, filterStatus, searchTerm, sortOrder)
    } catch (error) {
      console.error('Error loading history:', error)
      toast({
        title: "Failed to load history",
        description: "There was an error loading your try-on history. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }, [filterStatus, searchTerm, sortOrder, toast])

  // Apply filters to history data
  const applyFilters = useCallback((
    data: TryOnHistoryItem[], 
    status: string, 
    search: string, 
    sort: "newest" | "oldest"
  ) => {
    let result = [...data]
    
    // Filter by status
    if (status !== "all") {
      result = result.filter(item => item.status === status)
    }
    
    // Filter by search term
    if (search.trim() !== "") {
      const term = search.toLowerCase()
      result = result.filter(item => 
        item.product_name.toLowerCase().includes(term)
      )
    }
    
    // Sort by date
    result.sort((a, b) => {
      const dateA = a.created_at ? new Date(a.created_at) : new Date(0)
      const dateB = b.created_at ? new Date(b.created_at) : new Date(0)
      return sort === "newest" 
        ? dateB.getTime() - dateA.getTime() 
        : dateA.getTime() - dateB.getTime()
    })
    
    setFilteredHistory(result)
    setCurrentPage(1) // Reset to first page when filters change
    // Clear selections when filters change
    setSelectedItems([])
  }, [])

  // Update filters when inputs change
  useEffect(() => {
    applyFilters(history, filterStatus, searchTerm, sortOrder)
  }, [history, filterStatus, searchTerm, sortOrder, applyFilters])

  // Initial load and auto-refresh
  useEffect(() => {
    loadHistory()
    
    // Set up auto-refresh every minute in case new results are processed
    const refreshInterval = setInterval(() => {
      loadHistory()
    }, 60000) // 1 minute
    
    return () => clearInterval(refreshInterval)
  }, [loadHistory])

  // Toggle selection mode
  const toggleSelectionMode = () => {
    setIsSelectionMode(!isSelectionMode)
    if (isSelectionMode) {
      setSelectedItems([])
    }
  }

  // Handle selection of an item
  const toggleItemSelection = (id: string) => {
    if (selectedItems.includes(id)) {
      setSelectedItems(selectedItems.filter(itemId => itemId !== id))
    } else {
      setSelectedItems([...selectedItems, id])
    }
  }

  // Select all items on current page
  const selectAllCurrentPage = () => {
    const currentPageItems = paginatedHistory.map(item => item.id!)
    
    if (selectedItems.length === currentPageItems.length) {
      // If all are selected, deselect all
      setSelectedItems([])
    } else {
      // Otherwise select all on current page
      setSelectedItems(currentPageItems)
    }
  }

  // Handle bulk delete
  const handleBulkDelete = () => {
    if (selectedItems.length === 0) return
    setIsBulkDeleting(true)
  }

  // Confirm bulk delete
  const confirmBulkDelete = async () => {
    if (selectedItems.length === 0) return

    try {
      const { success, deletedCount } = await bulkDeleteHistoryItems(selectedItems)
      
      if (success) {
        toast({
          title: "Successfully deleted",
          description: `${deletedCount} try-on ${deletedCount === 1 ? 'result has' : 'results have'} been removed from your history`,
          variant: "default",
        })
        // Refresh the history list
        loadHistory()
      } else {
        toast({
          title: "Error",
          description: "Failed to remove try-on results. Please try again.",
          variant: "destructive",
        })
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "An unexpected error occurred",
        variant: "destructive",
      })
    } finally {
      setIsBulkDeleting(false)
      setSelectedItems([])
      setIsSelectionMode(false)
    }
  }

  // Handle delete single item
  const handleDelete = (id: string) => {
    setDeleteItemId(id)
  }

  // Confirm delete handler for single item
  const confirmDelete = async () => {
    if (!deleteItemId) return

    try {
      const success = await deleteHistoryItem(deleteItemId)
      if (success) {
        toast({
          title: "Successfully deleted",
          description: "The try-on result has been removed from your history",
          variant: "default",
        })
        // Refresh the history list
        loadHistory()
      } else {
        toast({
          title: "Error",
          description: "Failed to remove try-on result. Please try again.",
          variant: "destructive",
        })
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "An unexpected error occurred",
        variant: "destructive",
      })
    } finally {
      setDeleteItemId(null)
    }
  }

  // Cancel delete handler
  const cancelDelete = () => {
    setDeleteItemId(null)
  }

  // Cancel bulk delete handler
  const cancelBulkDelete = () => {
    setIsBulkDeleting(false)
  }

  // Handle image download
  const handleDownload = async (imageUrl: string, productName: string = "try-on-result") => {
    if (!imageUrl) return
    
    try {
      // Create a filename with a fallback in case productName is undefined
      const filename = productName 
        ? `virtual-tryon-${productName.replace(/\s+/g, '-').toLowerCase()}.jpg`
        : `virtual-tryon-result.jpg`;
      
      // Start loading indicator
      toast({
        title: "Preparing download",
        description: "Downloading your image...",
        variant: "default",
      })
      
      // Create a new Image element to load the image
      const img = new Image();
      img.crossOrigin = "anonymous"; // Try to avoid CORS issues
      
      // Set up a promise to wait for the image to load
      const imgLoaded = new Promise((resolve, reject) => {
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("Failed to load image"));
        img.src = imageUrl;
      });
      
      // Wait for the image to load
      const loadedImg = await imgLoaded;
      
      // Create a canvas element to draw the image
      const canvas = document.createElement('canvas');
      canvas.width = (loadedImg as HTMLImageElement).width;
      canvas.height = (loadedImg as HTMLImageElement).height;
      
      // Draw the image on the canvas
      const ctx = canvas.getContext('2d');
      ctx?.drawImage(loadedImg as HTMLImageElement, 0, 0);
      
      // Convert the canvas to a blob
      canvas.toBlob((blob) => {
        if (!blob) {
          throw new Error("Failed to create blob from image");
        }
        
        // Create a blob URL
        const blobUrl = URL.createObjectURL(blob);
        
        // Create a temporary link element
        const link = document.createElement('a');
        link.href = blobUrl;
        link.download = filename;
        link.style.display = 'none';
        document.body.appendChild(link);
        
        // Trigger download
        link.click();
        
        // Clean up
        document.body.removeChild(link);
        
        // Revoke the blob URL to free up memory
        setTimeout(() => {
          URL.revokeObjectURL(blobUrl);
        }, 100);
        
        toast({
          title: "Download complete",
          description: "Image has been saved to your device",
          variant: "default",
        });
      }, 'image/jpeg', 0.95);
    } catch (error) {
      console.error('Error downloading image:', error);
      
      // Fallback method if the canvas approach fails
      try {
        toast({
          title: "Trying alternative download method",
          description: "Please wait...",
          variant: "default",
        });
        
        // Create temporary iframe
        const iframe = document.createElement('iframe');
        iframe.style.display = 'none';
        document.body.appendChild(iframe);
        
        // Force download using iframe
        const iframeDoc = iframe.contentWindow?.document;
        if (iframeDoc) {
          iframeDoc.open();
          iframeDoc.write(`<a href="${imageUrl}" download="virtual-tryon.jpg" id="download-link">Download</a>`);
          iframeDoc.close();
          const downloadLink = iframeDoc.getElementById('download-link');
          if (downloadLink) {
            downloadLink.click();
            setTimeout(() => {
              document.body.removeChild(iframe);
            }, 1000);
            toast({
              title: "Download initiated",
              description: "Your browser should prompt you to save the image",
              variant: "default",
            });
          }
        }
      } catch (fallbackError) {
        console.error('Fallback download method failed:', fallbackError);
        toast({
          title: "Download failed",
          description: "Could not download the image. Please right-click and save manually.",
          variant: "destructive",
        });
      }
    }
  }

  // Handle sharing of try-on result
  const handleShare = (imageUrl: string, productName: string = "try-on-result") => {
    if (!imageUrl) return
    
    try {
      if (navigator.share) {
        navigator.share({
          title: `Virtual Try-On: ${productName || "Product"}`,
          text: `Check out how I look in this ${productName || "outfit"}!`,
          url: imageUrl,
        })
        .then(() => {
          toast({
            title: "Shared successfully",
            description: "Your try-on has been shared",
            variant: "default",
          })
        })
        .catch((error) => {
          console.error('Error sharing:', error)
          // Fallback to copy to clipboard
          copyToClipboard(imageUrl)
        })
      } else {
        // Fallback for browsers that don't support sharing
        copyToClipboard(imageUrl)
      }
    } catch (error) {
      console.error('Error in share functionality:', error)
      copyToClipboard(imageUrl)
    }
  }

  // Copy URL to clipboard
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(
      () => {
        toast({
          title: "Copied to clipboard",
          description: "Image URL has been copied to your clipboard",
          variant: "default",
        })
      },
      (err) => {
        console.error('Could not copy text: ', err)
        toast({
          title: "Failed to copy",
          description: "Could not copy URL to clipboard",
          variant: "destructive",
        })
      }
    )
  }

  // Calculate pagination
  const totalPages = Math.ceil(filteredHistory.length / itemsPerPage)
  const paginatedHistory = filteredHistory.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  )

  return (
    <div className="container py-10" role="main">
      <header className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <History className="h-6 w-6 text-primary" aria-hidden="true" />
          <h1 className="text-3xl font-bold">Your Virtual Try-On History</h1>
        </div>
        <div className="flex items-center gap-2">
          {isSelectionMode ? (
            <Button 
              variant="destructive" 
              size="sm" 
              onClick={handleBulkDelete}
              disabled={selectedItems.length === 0}
              className="gap-2"
            >
              <Trash2 className="h-4 w-4" aria-hidden="true" />
              Delete {selectedItems.length > 0 ? `(${selectedItems.length})` : ''}
            </Button>
          ) : (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="sm" onClick={loadHistory} className="gap-2">
                    <RefreshCw className="h-4 w-4" aria-hidden="true" />
                    <span className="sr-only md:not-sr-only">Refresh</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Refresh your try-on history</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
          
          <Button 
            variant={isSelectionMode ? "secondary" : "outline"} 
            size="sm" 
            onClick={toggleSelectionMode}
            className="gap-2"
          >
            {isSelectionMode ? "Cancel" : "Select"}
          </Button>
        </div>
      </header>

      {/* Filters and controls section */}
      <div className="flex flex-col md:flex-row gap-3 mb-6">
        <div className="relative flex-1">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" aria-hidden="true" />
          <Input
            placeholder="Search by product name..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-8"
            aria-label="Search try-on history"
          />
        </div>
        
        <div className="flex gap-2">
          <Select
            value={filterStatus}
            onValueChange={(value) => setFilterStatus(value)}
            aria-label="Filter by status"
          >
            <SelectTrigger className="w-[140px]">
              <Filter className="h-4 w-4 mr-2" aria-hidden="true" />
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="pending">Pending</SelectItem>
              <SelectItem value="processing">Processing</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
            </SelectContent>
          </Select>
          
          <Select
            value={sortOrder}
            onValueChange={(value) => setSortOrder(value as "newest" | "oldest")}
            aria-label="Sort by date"
          >
            <SelectTrigger className="w-[140px]">
              <ArrowDownUp className="h-4 w-4 mr-2" aria-hidden="true" />
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="newest">Newest First</SelectItem>
              <SelectItem value="oldest">Oldest First</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {isLoading ? (
        <div className="text-center py-12" aria-live="polite" aria-busy={isLoading}>
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto" aria-hidden="true"></div>
          <p className="mt-4 text-muted-foreground">Loading try-on history...</p>
        </div>
      ) : filteredHistory.length === 0 ? (
        <div 
          className="text-center py-12 border rounded-xl bg-secondary/20" 
          role="region" 
          aria-labelledby="no-history-title"
        >
          {searchTerm || filterStatus !== "all" ? (
            <>
              <h2 id="no-history-title" className="text-xl font-medium mb-4">No matching results</h2>
              <p className="text-muted-foreground mb-6">
                We couldn't find any try-on history matching your filters. Try adjusting your search criteria.
              </p>
              <Button 
                variant="outline" 
                onClick={() => {
                  setSearchTerm("")
                  setFilterStatus("all")
                }}
              >
                Clear Filters
              </Button>
            </>
          ) : (
            <>
              <h2 id="no-history-title" className="text-xl font-medium mb-4">No try-on history yet</h2>
              <p className="text-muted-foreground mb-6">
                You haven't tried on any clothes yet. Visit our collection and try on some outfits to see them here!
              </p>
              <Button asChild>
                <Link href="/categories">Browse Collection</Link>
              </Button>
            </>
          )}
        </div>
      ) : (
        <>
          {/* Bulk selection controls */}
          {isSelectionMode && (
            <div className="mb-4 p-2 bg-muted rounded-md flex items-center justify-between">
              <div className="flex items-center">
                <input 
                  type="checkbox" 
                  id="select-all"
                  className="h-4 w-4 mr-2 cursor-pointer" 
                  checked={selectedItems.length === paginatedHistory.length && paginatedHistory.length > 0}
                  onChange={selectAllCurrentPage}
                  aria-label="Select all items on current page"
                />
                <label htmlFor="select-all" className="text-sm cursor-pointer">
                  {selectedItems.length === 0 
                    ? "Select all on this page" 
                    : `Selected ${selectedItems.length} ${selectedItems.length === 1 ? 'item' : 'items'}`
                  }
                </label>
              </div>
              
              {selectedItems.length > 0 && (
                <Button 
                  variant="destructive" 
                  size="sm" 
                  onClick={handleBulkDelete}
                  className="gap-1"
                >
                  <Trash2 className="h-4 w-4" aria-hidden="true" />
                  Delete Selected
                </Button>
              )}
            </div>
          )}

          <div className="grid gap-6">
            {paginatedHistory.map((item) => (
              <div
                key={item.id}
                className={`border rounded-lg shadow-sm overflow-hidden ${
                  selectedItems.includes(item.id!) ? "ring-2 ring-primary" : ""
                }`}
              >
                <div className="flex justify-between items-center p-3 bg-muted/30 border-b">
                  <div className="flex items-center gap-2">
                    {isSelectionMode && (
                      <input 
                        type="checkbox" 
                        checked={selectedItems.includes(item.id!)}
                        onChange={() => toggleItemSelection(item.id!)}
                        className="h-4 w-4 mr-1 cursor-pointer"
                        aria-label={`Select ${item.product_name}`}
                      />
                    )}
                    <span className="font-medium">{item.product_name}</span>
                    {item.status === 'completed' && (
                      <Badge variant="success">Completed</Badge>
                    )}
                    {item.status === 'pending' && (
                      <Badge variant="default">Pending</Badge>
                    )}
                    {item.status === 'processing' && (
                      <Badge variant="secondary">Processing</Badge>
                    )}
                    {item.status === 'failed' && (
                      <Badge variant="destructive">Failed</Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    {!isSelectionMode && (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => handleDelete(item.id!)}
                              className="text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                              aria-label="Delete this try-on result"
                            >
                              <Trash2 className="h-4 w-4" aria-hidden="true" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent side="left">
                            <p>Delete this try-on result</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}
                  </div>
                </div>
                
                <div className="p-4 flex flex-col md:flex-row md:items-center gap-4">
                  <div className="flex-shrink-0 flex gap-2">
                    {/* User Photo */}
                    <div 
                      className="relative cursor-pointer w-[100px] h-[130px]"
                      onClick={() => !isSelectionMode && setPreviewImage(item.user_image_url)}
                    >
                      <Image
                        src={item.user_image_url || "/placeholder-user.jpg"}
                        alt={`Your uploaded photo for ${item.product_name} try-on`}
                        width={100}
                        height={130}
                        className={`rounded-md object-cover border border-gray-200 ${!isSelectionMode && "hover:border-primary"} transition-colors w-full h-full`}
                        loading="lazy"
                      />
                      <div className="absolute bottom-0 left-0 bg-black/70 text-white text-xs px-1 rounded-bl-md">
                        You
                      </div>
                    </div>
                    
                    {/* Clothing Item */}
                    <div 
                      className="relative cursor-pointer w-[100px] h-[130px]"
                      onClick={() => !isSelectionMode && setPreviewImage(item.cloth_image_url || item.product_image)}
                    >
                      <Image
                        src={item.cloth_image_url || item.product_image || "/placeholder.svg"}
                        alt={`Clothing item: ${item.product_name}`}
                        width={100}
                        height={130}
                        className={`rounded-md object-cover border border-gray-200 ${!isSelectionMode && "hover:border-primary"} transition-colors w-full h-full`}
                        loading="lazy"
                      />
                      <div className="absolute bottom-0 left-0 bg-black/70 text-white text-xs px-1 rounded-bl-md">
                        Item
                      </div>
                    </div>
                    
                    {/* Result Image */}
                    <div 
                      className="relative cursor-pointer w-[100px] h-[130px]"
                      onClick={() => !isSelectionMode && setPreviewImage(item.result_image_url)}
                    >
                      <Image
                        src={item.result_image_url || "/placeholder.svg"}
                        alt={`Virtual try-on result for ${item.product_name}`}
                        width={100}
                        height={130}
                        className={`rounded-md object-cover border border-gray-200 ${!isSelectionMode && "hover:border-primary"} transition-colors w-full h-full`}
                        loading="lazy"
                      />
                      <div className="absolute bottom-0 left-0 bg-black/70 text-white text-xs px-1 rounded-bl-md">
                        Result
                      </div>
                    </div>
                  </div>
                  <div className="flex-1">
                    {item.status === 'failed' && (
                      <p className="text-sm text-red-600 mt-1">{item.error || 'Processing failed'}</p>
                    )}
                  </div>
                  <div className="flex flex-col gap-2">
                    {!isSelectionMode && (
                      <>
                        <Button asChild variant="outline" size="sm" className="gap-1">
                          <Link href={`/try-on/${item.product_id}`}>
                            <Eye className="h-4 w-4" aria-hidden="true" />
                            Try Again
                          </Link>
                        </Button>
                        
                        <div className="flex gap-1">
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button 
                                variant="secondary" 
                                size="sm" 
                                disabled={!item.result_image_url || item.status !== 'completed'}
                                className="gap-1"
                              >
                                Actions
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem 
                                onClick={() => setPreviewImage(item.result_image_url)}
                                disabled={!item.result_image_url}
                              >
                                <Eye className="h-4 w-4 mr-2" aria-hidden="true" />
                                View Full Size
                              </DropdownMenuItem>                          <DropdownMenuItem 
                            onClick={() => handleDownload(item.result_image_url, item.product_name || "product")}
                            disabled={!item.result_image_url || item.status !== 'completed'}
                          >
                            <Download className="h-4 w-4 mr-2" aria-hidden="true" />
                            Download Image
                          </DropdownMenuItem>
                          <DropdownMenuItem 
                            onClick={() => handleShare(item.result_image_url, item.product_name || "product")}
                            disabled={!item.result_image_url || item.status !== 'completed'}
                          >
                            <Share2 className="h-4 w-4 mr-2" aria-hidden="true" />
                            Share Result
                          </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                          
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleDelete(item.id!)}
                            className="text-destructive hover:bg-destructive/10 hover:text-destructive border-destructive/30 flex items-center"
                          >
                            <Trash2 className="h-4 w-4 mr-1" aria-hidden="true" />
                            Remove
                          </Button>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="flex justify-center mt-6" role="navigation" aria-label="Pagination Navigation">
              <div className="flex items-center gap-1">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
                  disabled={currentPage === 1}
                  aria-label="Previous page"
                >
                  Previous
                </Button>
                
                <div className="flex items-center mx-2">
                  <span className="text-sm">
                    Page <strong>{currentPage}</strong> of <strong>{totalPages}</strong>
                  </span>
                </div>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
                  disabled={currentPage === totalPages}
                  aria-label="Next page"
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </>
      )}

      {/* Image Preview Modal */}
      {previewImage && (
        <Dialog open={!!previewImage} onOpenChange={() => setPreviewImage(null)}>
          <DialogContent className="sm:max-w-[90vw] sm:max-h-[90vh] overflow-auto p-1 flex flex-col">
            <DialogClose className="absolute top-2 right-2 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground z-10">
              <X className="h-6 w-6" aria-hidden="true" />
              <span className="sr-only">Close image preview</span>
            </DialogClose>
            
            {/* We add the DialogTitle but make it visually hidden for accessibility */}
            <DialogTitle className="sr-only">Image Preview</DialogTitle>
            
            <div className="relative w-full h-full min-h-[50vh] flex items-center justify-center overflow-hidden pt-6">
              <img 
                src={previewImage} 
                alt="Detailed preview of try-on image" 
                className="max-w-full max-h-[70vh] object-contain mx-auto"
                style={{ display: 'block' }}
              />
              <div className="absolute bottom-2 right-2 flex gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  className="gap-1"
                  onClick={() => handleDownload(previewImage)}
                >
                  <Download className="h-4 w-4" aria-hidden="true" />
                  Download
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  className="gap-1"
                  onClick={() => handleShare(previewImage, "try-on-result")}
                >
                  <Share2 className="h-4 w-4" aria-hidden="true" />
                  Share
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      )}
      
      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!deleteItemId} onOpenChange={(open) => !open && setDeleteItemId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Try-on Result</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently remove this try-on result from your history.
              Any saved images will remain on the server, but the record will be deleted from your account history.
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={cancelDelete}>Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={confirmDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Bulk Delete Confirmation Dialog */}
      <AlertDialog open={isBulkDeleting} onOpenChange={(open) => !open && setIsBulkDeleting(false)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Multiple Try-on Results</AlertDialogTitle>
            <AlertDialogDescription>
              You are about to delete {selectedItems.length} {selectedItems.length === 1 ? 'item' : 'items'} from your try-on history.
              This action cannot be undone. The images will remain on the server, but they will be removed from your history.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={cancelBulkDelete}>Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={confirmBulkDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete {selectedItems.length} {selectedItems.length === 1 ? 'Item' : 'Items'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
