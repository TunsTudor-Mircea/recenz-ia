"use client"

import { useEffect, useState, Suspense } from "react"
import { useRouter } from "next/navigation"
import { Navbar } from "@/components/layout/navbar"
import { AddProductModal } from "@/components/products/add-product-modal"
import { ProductCard } from "@/components/products/product-card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { useToast } from "@/hooks/use-toast"
import { Search, Package, ChevronLeft, ChevronRight } from "lucide-react"
import { isAuthenticated } from "@/lib/auth"
import api from "@/lib/api"
import type { SentimentDistribution } from "@/types/api"

interface ProductSummary {
  name: string
  total_reviews: number
  average_rating: number
  sentiment_distribution: SentimentDistribution
  last_updated: string | null
}

interface PaginatedProducts {
  products: ProductSummary[]
  total: number
  skip: number
  limit: number
}

function ProductsContent() {
  const [products, setProducts] = useState<ProductSummary[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [debouncedSearch, setDebouncedSearch] = useState("")
  const [sortBy, setSortBy] = useState("updated_at")
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc")
  const [currentPage, setCurrentPage] = useState(1)
  const [totalProducts, setTotalProducts] = useState(0)
  const productsPerPage = 6
  const router = useRouter()
  const { toast } = useToast()

  // Debounce search query
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchQuery)
      setCurrentPage(1) // Reset to first page on new search
    }, 500)

    return () => clearTimeout(timer)
  }, [searchQuery])

  const fetchProducts = async () => {
    try {
      setIsLoading(true)

      const skip = (currentPage - 1) * productsPerPage
      const params = new URLSearchParams({
        skip: skip.toString(),
        limit: productsPerPage.toString(),
        sort_by: sortBy,
        sort_order: sortOrder,
      })

      if (debouncedSearch) {
        params.append("search", debouncedSearch)
      }

      const response = await api.get<PaginatedProducts>(
        `/api/v1/products/?${params.toString()}`
      )

      setProducts(response.data.products)
      setTotalProducts(response.data.total)
    } catch (error) {
      console.error("Failed to fetch products:", error)
      toast({
        title: "Error",
        description: "Failed to load products. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (!isAuthenticated()) {
      router.push("/login")
      return
    }

    fetchProducts()
  }, [router, currentPage, sortBy, sortOrder, debouncedSearch])

  const totalPages = Math.ceil(totalProducts / productsPerPage)

  const handleSortChange = (value: string) => {
    // Parse sort value (format: "field-order")
    const [field, order] = value.split("-")
    setSortBy(field)
    setSortOrder(order as "asc" | "desc")
    setCurrentPage(1) // Reset to first page on sort change
  }

  const getCurrentSortValue = () => {
    return `${sortBy}-${sortOrder}`
  }

  const handleDeleteProduct = async (productName: string) => {
    try {
      await api.delete(`/api/v1/products/${encodeURIComponent(productName)}`)

      toast({
        title: "Success",
        description: `Product "${productName}" has been deleted successfully.`,
      })

      // Refresh the products list
      fetchProducts()
    } catch (error) {
      console.error("Failed to delete product:", error)
      toast({
        title: "Error",
        description: "Failed to delete product. Please try again.",
        variant: "destructive",
      })
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto p-4 md:p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Products</h1>
            <p className="text-muted-foreground mt-1">Manage and analyze your product reviews</p>
          </div>
          <AddProductModal onSuccess={() => {
            setCurrentPage(1)
            fetchProducts()
          }} />
        </div>

        {/* Filters */}
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search products..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
          <Select value={getCurrentSortValue()} onValueChange={handleSortChange}>
            <SelectTrigger className="w-full sm:w-56">
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="name-asc">Name (A-Z)</SelectItem>
              <SelectItem value="name-desc">Name (Z-A)</SelectItem>
              <SelectItem value="rating-desc">Highest Rating</SelectItem>
              <SelectItem value="rating-asc">Lowest Rating</SelectItem>
              <SelectItem value="reviews-desc">Most Reviews</SelectItem>
              <SelectItem value="reviews-asc">Least Reviews</SelectItem>
              <SelectItem value="updated_at-desc">Recently Updated</SelectItem>
              <SelectItem value="updated_at-asc">Oldest Updated</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Results Count */}
        {!isLoading && (
          <div className="text-sm text-muted-foreground">
            Showing {products.length > 0 ? ((currentPage - 1) * productsPerPage) + 1 : 0} to {Math.min(currentPage * productsPerPage, totalProducts)} of {totalProducts} products
          </div>
        )}

        {/* Products Grid */}
        {isLoading ? (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3">
            {[...Array(6)].map((_, i) => (
              <Skeleton key={i} className="h-80" />
            ))}
          </div>
        ) : products.length > 0 ? (
          <>
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3">
              {products.map((product) => (
                <ProductCard
                  key={product.name}
                  name={product.name}
                  totalReviews={product.total_reviews}
                  averageRating={product.average_rating}
                  sentimentDistribution={product.sentiment_distribution}
                  lastUpdated={product.last_updated || new Date().toISOString()}
                  onDelete={handleDeleteProduct}
                />
              ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-8">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                  disabled={currentPage === 1}
                >
                  <ChevronLeft className="h-4 w-4 mr-1" />
                  Previous
                </Button>

                <div className="flex items-center gap-1">
                  {[...Array(Math.min(totalPages, 7))].map((_, idx) => {
                    let pageNum: number

                    if (totalPages <= 7) {
                      pageNum = idx + 1
                    } else if (currentPage <= 4) {
                      pageNum = idx + 1
                    } else if (currentPage >= totalPages - 3) {
                      pageNum = totalPages - 6 + idx
                    } else {
                      pageNum = currentPage - 3 + idx
                    }

                    if (pageNum < 1 || pageNum > totalPages) return null

                    return (
                      <Button
                        key={pageNum}
                        variant={currentPage === pageNum ? "default" : "outline"}
                        size="sm"
                        onClick={() => setCurrentPage(pageNum)}
                        className="w-10"
                      >
                        {pageNum}
                      </Button>
                    )
                  })}
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                  disabled={currentPage === totalPages}
                >
                  Next
                  <ChevronRight className="h-4 w-4 ml-1" />
                </Button>
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Package className="h-16 w-16 text-muted-foreground/50 mb-4" />
            <h3 className="text-lg font-semibold mb-2">No products found</h3>
            <p className="text-muted-foreground mb-4">
              {searchQuery ? "Try adjusting your search" : "Get started by adding your first product"}
            </p>
            {!searchQuery && <AddProductModal onSuccess={() => {
              setCurrentPage(1)
              fetchProducts()
            }} />}
          </div>
        )}
      </main>
    </div>
  )
}

export default function ProductsPage() {
  return (
    <Suspense fallback={null}>
      <ProductsContent />
    </Suspense>
  )
}
