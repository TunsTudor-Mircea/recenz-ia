"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Navbar } from "@/components/layout/navbar"
import { StatCard } from "@/components/dashboard/stat-card"
import { SentimentDistributionChart } from "@/components/dashboard/sentiment-distribution-chart"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Package, MessageSquare, Star, TrendingUp, ArrowRight } from "lucide-react"
import { isAuthenticated, getCurrentUser } from "@/lib/auth"
import api from "@/lib/api"
import type { AnalyticsSummary, AspectAnalytics } from "@/types/api"
import { ASPECT_LABELS } from "@/components/charts/aspect-sentiment-chart"
import { Skeleton } from "@/components/ui/skeleton"
import { useToast } from "@/hooks/use-toast"
import Link from "next/link"

function healthLabel(rate: number): { label: string; className: string } {
  if (rate >= 80) return { label: "Excellent", className: "bg-green-100 text-green-700" }
  if (rate >= 60) return { label: "Good", className: "bg-teal-100 text-teal-700" }
  if (rate >= 40) return { label: "Fair", className: "bg-yellow-100 text-yellow-700" }
  return { label: "Poor", className: "bg-red-100 text-red-700" }
}

export default function DashboardPage() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null)
  const [aspectAnalytics, setAspectAnalytics] = useState<AspectAnalytics | null>(null)
  const [userName, setUserName] = useState<string>("")
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()
  const { toast } = useToast()

  useEffect(() => {
    if (!isAuthenticated()) {
      router.push("/login")
      return
    }

    const fetchSummary = async () => {
      try {
        const [summaryResponse, userResponse] = await Promise.all([
          api.get<AnalyticsSummary>("/api/v1/analytics/summary"),
          getCurrentUser(),
        ])
        setSummary(summaryResponse.data)
        setUserName(userResponse.full_name || userResponse.email)

        // Best-effort: fetch aspect analytics (only available if ABSA reviews exist)
        try {
          const aspectResponse = await api.get<AspectAnalytics>("/api/v1/analytics/aspects")
          setAspectAnalytics(aspectResponse.data)
        } catch {
          // no ABSA data yet — silently ignore
        }
      } catch (error) {
        console.error("[v0] Failed to fetch summary:", error)
        toast({
          title: "Error",
          description: "Failed to load analytics summary. Please try again.",
          variant: "destructive",
        })
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

  const positive = summary?.sentiment_distribution.positive ?? 0
  const negative = summary?.sentiment_distribution.negative ?? 0
  const total = summary?.sentiment_distribution.total ?? 0
  const positiveRate = total > 0 ? Math.round((positive / total) * 100) : 0
  const negativeRate = total > 0 ? Math.round((negative / total) * 100) : 0
  const avgRating = summary?.average_rating ?? 0
  const health = healthLabel(positiveRate)

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
          {userName && (
            <p className="text-lg font-medium text-muted-foreground">
              Hello, <span className="text-foreground font-semibold">{userName.split(" ")[0]}</span>
            </p>
          )}
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
            value={`${positiveRate}%`}
            icon={TrendingUp}
            description="Overall sentiment score"
          />
        </div>

        {/* Row 1: Sentiment chart + Sentiment Overview */}
        <div className="grid gap-6 lg:grid-cols-2">
          {summary && <SentimentDistributionChart data={summary.sentiment_distribution} />}

          <Card className="flex flex-col">
            <CardHeader>
              <CardTitle>Sentiment Overview</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col flex-1 gap-5">
              {/* Big positive rate */}
              <div className="flex items-end gap-3">
                <span className="text-6xl font-bold" style={{ color: "#14b8a6" }}>
                  {positiveRate}%
                </span>
                <div className="pb-2 space-y-0.5">
                  <p className="text-sm font-semibold text-foreground">positive</p>
                  <p className="text-xs text-muted-foreground">out of {total} reviews</p>
                </div>
              </div>

              {/* Stacked bar */}
              <div className="space-y-1.5">
                <div className="flex h-2.5 w-full rounded-full overflow-hidden bg-muted">
                  <div
                    style={{ width: `${positiveRate}%`, backgroundColor: "#14b8a6", transition: "width 0.5s ease" }}
                  />
                  <div
                    style={{ width: `${negativeRate}%`, backgroundColor: "#475569" }}
                  />
                </div>
                <div className="flex justify-between text-xs">
                  <span style={{ color: "#14b8a6" }} className="font-medium">{positive} positive</span>
                  <span style={{ color: "#475569" }} className="font-medium">{negative} negative</span>
                </div>
              </div>

              <div className="flex-1" />

              {/* Average rating + health badge */}
              <div className="pt-2 border-t space-y-1.5">
                <p className="text-xs text-muted-foreground uppercase tracking-wide">Average Rating</p>
                <div className="flex items-center gap-2 flex-wrap">
                  <div className="flex items-center gap-0.5">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <Star
                        key={star}
                        className="h-4 w-4"
                        fill={star <= Math.round(avgRating) ? "#f59e0b" : "none"}
                        stroke={star <= Math.round(avgRating) ? "#f59e0b" : "#d1d5db"}
                      />
                    ))}
                  </div>
                  <span className="text-lg font-semibold">{avgRating.toFixed(1)}</span>
                  <span className="text-xs text-muted-foreground">/ 5</span>
                  {total > 0 && (
                    <span className={`ml-1 px-3 py-1 rounded-full text-sm font-semibold ${health.className}`}>
                      {health.label}
                    </span>
                  )}
                </div>
              </div>

              {/* Top aspects */}
              {aspectAnalytics && aspectAnalytics.aspects.length > 0 && (
                <div className="pt-2 border-t space-y-2">
                  <p className="text-xs text-muted-foreground uppercase tracking-wide">Top Aspects</p>
                  <div className="flex flex-wrap gap-1.5">
                    {aspectAnalytics.aspects
                      .filter((a) => a.positive + a.negative + a.neutral > 0)
                      .sort((a, b) => (b.positive + b.negative + b.neutral) - (a.positive + a.negative + a.neutral))
                      .slice(0, 5)
                      .map((a) => {
                        const mentioned = a.positive + a.negative + a.neutral
                        const dominant = a.positive >= a.negative ? "positive" : "negative"
                        const style = dominant === "positive"
                          ? "bg-teal-50 text-teal-700 border border-teal-200"
                          : "bg-slate-100 text-slate-600 border border-slate-200"
                        return (
                          <span key={a.aspect} className={`text-xs font-medium px-2.5 py-1 rounded-full ${style}`}>
                            {ASPECT_LABELS[a.aspect] ?? a.aspect}
                            <span className="ml-1 opacity-60">{mentioned}</span>
                          </span>
                        )
                      })}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Row 2: Top Rated + Most Reviewed */}
        <div className="grid gap-6 lg:grid-cols-2">
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
                      <div className="flex items-center gap-3 min-w-0">
                        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[var(--sentiment-positive-bg)] text-[var(--sentiment-positive)] font-semibold text-sm">
                          {index + 1}
                        </div>
                        <span className="font-medium group-hover:text-primary truncate">{product}</span>
                      </div>
                      <ArrowRight className="h-4 w-4 shrink-0 ml-2 text-muted-foreground group-hover:text-primary" />
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
                      <div className="flex items-center gap-3 min-w-0">
                        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary font-semibold text-sm">
                          {index + 1}
                        </div>
                        <span className="font-medium group-hover:text-primary truncate">{product}</span>
                      </div>
                      <ArrowRight className="h-4 w-4 shrink-0 ml-2 text-muted-foreground group-hover:text-primary" />
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
