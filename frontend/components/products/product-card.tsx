import Link from "next/link"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { StarRating } from "@/components/star-rating"
import { Badge } from "@/components/ui/badge"
import { MessageSquare, TrendingUp, Calendar } from "lucide-react"
import type { SentimentDistribution } from "@/types/api"

interface ProductCardProps {
  name: string
  totalReviews: number
  averageRating: number
  sentimentDistribution: SentimentDistribution
  lastUpdated?: string
}

export function ProductCard({
  name,
  totalReviews,
  averageRating,
  sentimentDistribution,
  lastUpdated,
}: ProductCardProps) {
  const positivePercentage = Math.round((sentimentDistribution.positive / sentimentDistribution.total) * 100)

  return (
    <Link href={`/products/${encodeURIComponent(name)}`}>
      <Card className="hover:shadow-lg transition-all duration-200 hover:border-primary/50 group h-full">
        <CardHeader>
          <CardTitle className="line-clamp-2 group-hover:text-primary transition-colors">{name}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <StarRating rating={averageRating} size={18} />
          </div>

          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="p-2 rounded-lg bg-[var(--sentiment-positive-bg)]">
              <div className="text-lg font-bold text-[var(--sentiment-positive)]">{sentimentDistribution.positive}</div>
              <div className="text-xs text-muted-foreground">Positive</div>
            </div>
            <div className="p-2 rounded-lg bg-[var(--sentiment-neutral-bg)]">
              <div className="text-lg font-bold text-[var(--sentiment-neutral)]">{sentimentDistribution.neutral}</div>
              <div className="text-xs text-muted-foreground">Neutral</div>
            </div>
            <div className="p-2 rounded-lg bg-[var(--sentiment-negative-bg)]">
              <div className="text-lg font-bold text-[var(--sentiment-negative)]">{sentimentDistribution.negative}</div>
              <div className="text-xs text-muted-foreground">Negative</div>
            </div>
          </div>

          <div className="flex items-center justify-between text-sm text-muted-foreground pt-2 border-t">
            <div className="flex items-center gap-1">
              <MessageSquare className="h-4 w-4" />
              <span>{totalReviews} reviews</span>
            </div>
            <Badge variant="outline" className="gap-1">
              <TrendingUp className="h-3 w-3" />
              {positivePercentage}%
            </Badge>
          </div>

          {lastUpdated && (
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <Calendar className="h-3 w-3" />
              <span>Updated {new Date(lastUpdated).toLocaleDateString()}</span>
            </div>
          )}
        </CardContent>
      </Card>
    </Link>
  )
}
