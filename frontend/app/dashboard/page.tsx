"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Navbar } from "@/components/layout/navbar"
import { StatCard } from "@/components/dashboard/stat-card"
import { SentimentDistributionChart } from "@/components/dashboard/sentiment-distribution-chart"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Package, MessageSquare, Star, TrendingUp, ArrowRight } from "lucide-react"
import { isAuthenticated } from "@/lib/auth"
import api from "@/lib/api"
import type { AnalyticsSummary } from "@/types/api"
import { Skeleton } from "@/components/ui/skeleton"
import Link from "next/link"

export default function DashboardPage() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    if (!isAuthenticated()) {
      router.push("/login")
      return
    }

    const fetchSummary = async () => {
      try {
        const response = await api.get<AnalyticsSummary>("/api/v1/analytics/summary")
        setSummary(response.data)
      } catch (error) {
        console.error("[v0] Failed to fetch summary:", error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchSummary()
  }, [router])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar />
        <main className="container mx-auto p-4 md:p-6 space-y-6">
          <div className="flex items-center justify-between">
            <Skeleton className="h-8 w-48" />
            <Skeleton className="h-10 w-32" />
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-32" />
            ))}
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto p-4 md:p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
            <p className="text-muted-foreground mt-1">Overview of your sentiment analysis data</p>
          </div>
          <Link href="/products">
            <Button>
              <Package className="h-4 w-4 mr-2" />
              View Products
            </Button>
          </Link>
        </div>

        {/* Stats Cards */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <StatCard
            title="Total Products"
            value={summary?.total_products || 0}
            icon={Package}
            description="Products being tracked"
          />
          <StatCard
            title="Total Reviews"
            value={summary?.total_reviews || 0}
            icon={MessageSquare}
            description="Reviews analyzed"
          />
          <StatCard
            title="Average Rating"
            value={summary?.average_rating.toFixed(1) || "0.0"}
            icon={Star}
            description="Across all products"
          />
          <StatCard
            title="Positive Sentiment"
            value={`${summary ? Math.round((summary.sentiment_distribution.positive / summary.sentiment_distribution.total) * 100) : 0}%`}
            icon={TrendingUp}
            description="Overall sentiment score"
          />
        </div>

        {/* Charts and Lists */}
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Sentiment Chart */}
          {summary && <SentimentDistributionChart data={summary.sentiment_distribution} />}

          {/* Top Rated Products */}
          <Card>
            <CardHeader>
              <CardTitle>Top Rated Products</CardTitle>
            </CardHeader>
            <CardContent>
              {summary && summary.top_rated_products.length > 0 ? (
                <div className="space-y-3">
                  {summary.top_rated_products.slice(0, 5).map((product, index) => (
                    <Link
                      key={index}
                      href={`/products/${encodeURIComponent(product)}`}
                      className="flex items-center justify-between p-3 rounded-lg border hover:bg-accent transition-colors group"
                    >
                      <div className="flex items-center gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[var(--sentiment-positive-bg)] text-[var(--sentiment-positive)] font-semibold text-sm">
                          {index + 1}
                        </div>
                        <span className="font-medium group-hover:text-primary">{product}</span>
                      </div>
                      <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-primary" />
                    </Link>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">No products yet</p>
              )}
            </CardContent>
          </Card>

          {/* Most Reviewed Products */}
          <Card>
            <CardHeader>
              <CardTitle>Most Reviewed Products</CardTitle>
            </CardHeader>
            <CardContent>
              {summary && summary.most_reviewed_products.length > 0 ? (
                <div className="space-y-3">
                  {summary.most_reviewed_products.slice(0, 5).map((product, index) => (
                    <Link
                      key={index}
                      href={`/products/${encodeURIComponent(product)}`}
                      className="flex items-center justify-between p-3 rounded-lg border hover:bg-accent transition-colors group"
                    >
                      <div className="flex items-center gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-primary font-semibold text-sm">
                          {index + 1}
                        </div>
                        <span className="font-medium group-hover:text-primary">{product}</span>
                      </div>
                      <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-primary" />
                    </Link>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">No products yet</p>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
