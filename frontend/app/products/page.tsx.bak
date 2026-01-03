"use client"

import { useEffect, useState, Suspense } from "react"
import { useRouter } from "next/navigation"
import { Navbar } from "@/components/layout/navbar"
import { AddProductModal } from "@/components/products/add-product-modal"
import { ProductCard } from "@/components/products/product-card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { useToast } from "@/hooks/use-toast"
import { Search, Package } from "lucide-react"
import { isAuthenticated } from "@/lib/auth"
import api from "@/lib/api"
import type { Review, SentimentDistribution } from "@/types/api"

interface ProductSummary {
  name: string
  totalReviews: number
  averageRating: number
  sentimentDistribution: SentimentDistribution
  lastUpdated: string
}

function ProductsContent() {
  const [products, setProducts] = useState<ProductSummary[]>([])
  const [filteredProducts, setFilteredProducts] = useState<ProductSummary[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [sortBy, setSortBy] = useState("name")
  const router = useRouter()
  const { toast } = useToast()

  const fetchProducts = async () => {
    try {
      setIsLoading(true)

      // Backend has a max limit of 100, so we need to fetch in batches
      let allReviews: Review[] = []
      let skip = 0
      const limit = 100
      let hasMore = true

      while (hasMore) {
        const response = await api.get<{ reviews: Review[]; total: number }>(
          `/api/v1/reviews/?skip=${skip}&limit=${limit}`
        )
        allReviews = [...allReviews, ...response.data.reviews]
        skip += limit
        hasMore = response.data.reviews.length === limit
      }

      // Group by product name and aggregate stats
      const productMap = new Map<string, ProductSummary>()

      allReviews.forEach((review) => {
        const existing = productMap.get(review.product_name)

        if (existing) {
          existing.totalReviews++
          existing.averageRating =
            (existing.averageRating * (existing.totalReviews - 1) + review.rating) / existing.totalReviews
          existing.sentimentDistribution[review.sentiment_label]++
          existing.sentimentDistribution.total++
          if (new Date(review.review_date) > new Date(existing.lastUpdated)) {
            existing.lastUpdated = review.review_date
          }
        } else {
          productMap.set(review.product_name, {
            name: review.product_name,
            totalReviews: 1,
            averageRating: review.rating,
            sentimentDistribution: {
              positive: review.sentiment_label === "positive" ? 1 : 0,
              neutral: review.sentiment_label === "neutral" ? 1 : 0,
              negative: review.sentiment_label === "negative" ? 1 : 0,
              total: 1,
            },
            lastUpdated: review.review_date,
          })
        }
      })

      const productsList = Array.from(productMap.values())
      setProducts(productsList)
      setFilteredProducts(productsList)
    } catch (error) {
      console.error("[v0] Failed to fetch products:", error)
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
  }, [router])

  useEffect(() => {
    let filtered = [...products]

    // Search filter
    if (searchQuery) {
      filtered = filtered.filter((product) => product.name.toLowerCase().includes(searchQuery.toLowerCase()))
    }

    // Sort
    filtered.sort((a, b) => {
      switch (sortBy) {
        case "name":
          return a.name.localeCompare(b.name)
        case "rating":
          return b.averageRating - a.averageRating
        case "reviews":
          return b.totalReviews - a.totalReviews
        case "date":
          return new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime()
        default:
          return 0
      }
    })

    setFilteredProducts(filtered)
  }, [searchQuery, sortBy, products])

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
          <AddProductModal onSuccess={fetchProducts} />
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
          <Select value={sortBy} onValueChange={setSortBy}>
            <SelectTrigger className="w-full sm:w-48">
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="name">Name (A-Z)</SelectItem>
              <SelectItem value="rating">Highest Rating</SelectItem>
              <SelectItem value="reviews">Most Reviews</SelectItem>
              <SelectItem value="date">Recently Updated</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Products Grid */}
        {isLoading ? (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {[...Array(6)].map((_, i) => (
              <Skeleton key={i} className="h-80" />
            ))}
          </div>
        ) : filteredProducts.length > 0 ? (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {filteredProducts.map((product) => (
              <ProductCard key={product.name} {...product} />
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Package className="h-16 w-16 text-muted-foreground/50 mb-4" />
            <h3 className="text-lg font-semibold mb-2">No products found</h3>
            <p className="text-muted-foreground mb-4">
              {searchQuery ? "Try adjusting your search" : "Get started by adding your first product"}
            </p>
            {!searchQuery && <AddProductModal onSuccess={fetchProducts} />}
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
