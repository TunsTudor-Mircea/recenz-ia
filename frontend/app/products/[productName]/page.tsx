"use client"

import { useEffect, useState, use, useMemo } from "react"
import { useRouter } from "next/navigation"
import { Navbar } from "@/components/layout/navbar"
import { StarRating } from "@/components/star-rating"
import { SentimentDistributionChart } from "@/components/dashboard/sentiment-distribution-chart"
import { RatingDistributionChart } from "@/components/charts/rating-distribution-chart"
import { SentimentTrendChart } from "@/components/charts/sentiment-trend-chart"
import { ReviewCard } from "@/components/reviews/review-card"
import { ReviewFilters } from "@/components/reviews/review-filters"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { RefreshCw, ArrowLeft, MessageSquare, BarChart2 } from "lucide-react"
import { isAuthenticated } from "@/lib/auth"
import { useToast } from "@/hooks/use-toast"
import api from "@/lib/api"
import type { ProductAnalytics, ProductTrend, Review, PaginatedReviews, AspectPolarityDistribution } from "@/types/api"
import { AspectSentimentChart, ASPECT_LABELS } from "@/components/charts/aspect-sentiment-chart"
import Link from "next/link"

export default function ProductDetailPage({ params }: { params: Promise<{ productName: string }> }) {
  const resolvedParams = use(params)
  const productName = decodeURIComponent(resolvedParams.productName)
  const [analytics, setAnalytics] = useState<ProductAnalytics | null>(null)
  const [trend, setTrend] = useState<ProductTrend | null>(null)
  const [allReviews, setAllReviews] = useState<Review[]>([])
  const [reviews, setReviews] = useState<Review[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)

  // Filters
  const [sentiment, setSentiment] = useState("all")
  const [rating, setRating] = useState("all")
  const [sortBy, setSortBy] = useState("date_desc")
  const [activeTab, setActiveTab] = useState("all")

  const router = useRouter()
  const { toast } = useToast()

  const fetchProductData = async () => {
    try {
      setIsLoading(true)

      // Fetch analytics
      const analyticsResponse = await api.get<ProductAnalytics>(
        `/api/v1/analytics/products/${encodeURIComponent(productName)}`,
      )
      setAnalytics(analyticsResponse.data)

      // Fetch trend
      const trendResponse = await api.get<ProductTrend>(
        `/api/v1/analytics/products/${encodeURIComponent(productName)}/trend?period=day&days=30`,
      )
      setTrend(trendResponse.data)

      // Fetch all reviews for client-side aspect computation (unfiltered)
      const allReviewsResponse = await api.get<PaginatedReviews>(
        `/api/v1/reviews/?product_name=${encodeURIComponent(productName)}&skip=0&limit=100`,
      )
      setAllReviews(allReviewsResponse.data.reviews)

      // Fetch reviews
      await fetchReviews()
    } catch (error) {
      console.error("[v0] Failed to fetch product data:", error)
      toast({
        title: "Error",
        description: "Failed to load product data",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const fetchReviews = async () => {
    try {
      let url = `/api/v1/reviews/?product_name=${encodeURIComponent(productName)}&skip=0&limit=50`

      if (sentiment !== "all") {
        url += `&sentiment_label=${sentiment}`
      }
      if (rating !== "all") {
        url += `&min_rating=${rating}`
      }

      const [sortField, sortOrder] = sortBy.split("_")
      url += `&sort_by=${sortField === "date" ? "review_date" : sortField === "sentiment" ? "sentiment_score" : sortField}`
      url += `&order=${sortOrder}`

      const reviewsResponse = await api.get<PaginatedReviews>(url)
      setReviews(reviewsResponse.data.reviews)
    } catch (error) {
      console.error("[v0] Failed to fetch reviews:", error)
    }
  }

  const handleRescan = async () => {
    setIsRefreshing(true)
    toast({
      title: "Re-scan Started",
      description: "Product reviews are being re-analyzed. This may take a few minutes.",
    })
    // In a real app, you'd trigger a new scraping job here
    setTimeout(() => {
      setIsRefreshing(false)
      fetchProductData()
    }, 3000)
  }

  useEffect(() => {
    if (!isAuthenticated()) {
      router.push("/login")
      return
    }

    fetchProductData()
  }, [router, productName])

  useEffect(() => {
    if (!isLoading) {
      fetchReviews()
    }
  }, [sentiment, rating, sortBy, activeTab])

  const aspectData = useMemo(() => {
    const absaReviews = allReviews.filter(
      (r) => r.model_used.startsWith("absa_") && r.aspects != null,
    )
    if (absaReviews.length === 0) return null

    const ALL_ASPECTS = [
      "BATERIE", "ECRAN", "SUNET", "PERFORMANTA", "CONECTIVITATE",
      "DESIGN", "CALITATE_CONSTRUCTIE", "PRET", "LIVRARE", "GENERAL",
    ]
    const counts: Record<string, { positive: number; negative: number; neutral: number; not_mentioned: number }> = {}
    for (const aspect of ALL_ASPECTS) {
      counts[aspect] = { positive: 0, negative: 0, neutral: 0, not_mentioned: 0 }
    }
    for (const review of absaReviews) {
      for (const [aspect, polarity] of Object.entries(review.aspects ?? {})) {
        if (!counts[aspect]) counts[aspect] = { positive: 0, negative: 0, neutral: 0, not_mentioned: 0 }
        if (polarity === "positive") counts[aspect].positive++
        else if (polarity === "negative") counts[aspect].negative++
        else if (polarity === "neutral") counts[aspect].neutral++
        else counts[aspect].not_mentioned++
      }
    }
    const aspects: AspectPolarityDistribution[] = ALL_ASPECTS.map((aspect) => ({
      aspect,
      ...counts[aspect],
      total_reviews: absaReviews.length,
    }))
    return { absa_count: absaReviews.length, aspects }
  }, [allReviews])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto p-4 md:p-6 space-y-6">
          <Skeleton className="h-12 w-3/4" />
          <div className="grid gap-6 lg:grid-cols-3">
            <Skeleton className="h-64" />
            <Skeleton className="h-64" />
            <Skeleton className="h-64" />
          </div>
        </main>
      </div>
    )
  }

  const topReviews = {
    positive: reviews
      .filter((r) => r.sentiment_label === "positive")
      .sort((a, b) => b.sentiment_score - a.sentiment_score)
      .slice(0, 5),
    negative: reviews
      .filter((r) => r.sentiment_label === "negative")
      .sort((a, b) => b.sentiment_score - a.sentiment_score)
      .slice(0, 5),
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto p-4 md:p-6 space-y-6">
        {/* Header */}
        <div className="space-y-4">
          <Link href="/products">
            <Button variant="ghost" size="sm">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Products
            </Button>
          </Link>

          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="space-y-2">
              <h1 className="text-3xl font-bold tracking-tight">{productName}</h1>
              <div className="flex items-center gap-4">
                <StarRating rating={analytics?.average_rating || 0} size={20} />
                <span className="text-muted-foreground">{analytics?.total_reviews || 0} reviews</span>
              </div>
            </div>
            <Button onClick={handleRescan} disabled={isRefreshing}>
              <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? "animate-spin" : ""}`} />
              Re-scan
            </Button>
          </div>
        </div>

        {/* Charts — all three on one row, wrap on smaller screens */}
        <div className="grid gap-6 lg:grid-cols-3">
          {analytics && <SentimentDistributionChart data={analytics.sentiment_distribution} />}
          {analytics && <RatingDistributionChart data={analytics.rating_distribution} />}
          {trend && <SentimentTrendChart data={trend} />}
        </div>

        {/* Aspect Analysis Section — computed from this product's own ABSA reviews */}
        {aspectData && (
          <div className="space-y-4">
            <div>
              <h2 className="text-xl font-semibold flex items-center gap-2">
                <BarChart2 className="h-5 w-5" />
                Aspect Analysis
              </h2>
              <p className="text-sm text-muted-foreground mt-1">
                Based on {aspectData.absa_count} reviews from this product
              </p>
            </div>

            {/* Chart + scrollable aspect list side by side on wide screens */}
            <div className="flex flex-col xl:flex-row gap-4 xl:h-[440px]">
              <div className="flex-1 min-w-0 xl:h-full">
                <AspectSentimentChart data={aspectData.aspects} />
              </div>

              <Card className="xl:w-80 xl:flex-shrink-0 w-full flex flex-col xl:h-full overflow-hidden">
                <CardHeader className="pb-2 pt-4 px-4">
                  <CardTitle className="text-sm font-semibold">By Aspect</CardTitle>
                </CardHeader>
                <CardContent className="px-3 pb-3 flex-1 overflow-y-auto min-h-0">
                  <div className="space-y-2 pr-1">
                    {aspectData.aspects
                      .filter((a) => a.positive + a.negative + a.neutral > 0)
                      .sort((a, b) => (b.positive + b.negative + b.neutral) - (a.positive + a.negative + a.neutral))
                      .map((a) => {
                        const mentioned = a.positive + a.negative + a.neutral
                        const mentionRate = Math.round((mentioned / aspectData.absa_count) * 100)
                        return (
                          <div key={a.aspect} className="rounded-md border px-3 py-2.5 space-y-1">
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-sm font-semibold">{ASPECT_LABELS[a.aspect] ?? a.aspect}</span>
                              <span className="text-xs text-muted-foreground shrink-0">{mentionRate}%</span>
                            </div>
                            <div className="text-xs flex items-center gap-1">
                              <span className="text-green-600 font-medium">{a.positive} pos</span>
                              <span className="text-muted-foreground">·</span>
                              <span className="text-red-600 font-medium">{a.negative} neg</span>
                              {a.neutral > 0 && (
                                <>
                                  <span className="text-muted-foreground">·</span>
                                  <span className="text-amber-600 font-medium">{a.neutral} neu</span>
                                </>
                              )}
                            </div>
                          </div>
                        )
                      })}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* Reviews Section */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5" />
                Reviews
              </CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="all">All Reviews</TabsTrigger>
                <TabsTrigger value="positive">Top Positive</TabsTrigger>
                <TabsTrigger value="negative">Top Negative</TabsTrigger>
              </TabsList>

              <TabsContent value="all" className="space-y-4">
                <ReviewFilters
                  sentiment={sentiment}
                  setSentiment={setSentiment}
                  rating={rating}
                  setRating={setRating}
                  sortBy={sortBy}
                  setSortBy={setSortBy}
                />
                <div className="grid gap-4 md:grid-cols-2">
                  {reviews.length > 0 ? (
                    reviews.map((review) => <ReviewCard key={review.id} review={review} />)
                  ) : (
                    <p className="text-center text-muted-foreground py-8">No reviews found</p>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="positive" className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  {topReviews.positive.length > 0 ? (
                    topReviews.positive.map((review) => <ReviewCard key={review.id} review={review} />)
                  ) : (
                    <p className="text-center text-muted-foreground py-8">No positive reviews yet</p>
                  )}
                </div>
              </TabsContent>

              <TabsContent value="negative" className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  {topReviews.negative.length > 0 ? (
                    topReviews.negative.map((review) => <ReviewCard key={review.id} review={review} />)
                  ) : (
                    <p className="text-center text-muted-foreground py-8">No negative reviews yet</p>
                  )}
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
