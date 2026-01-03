"use client"

import Link from "next/link"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { StarRating } from "@/components/star-rating"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { MessageSquare, TrendingUp, Calendar, Trash2 } from "lucide-react"
import type { SentimentDistribution } from "@/types/api"

interface ProductCardProps {
  name: string
  totalReviews: number
  averageRating: number
  sentimentDistribution: SentimentDistribution
  lastUpdated?: string
  onDelete?: (productName: string) => void
}

export function ProductCard({
  name,
  totalReviews,
  averageRating,
  sentimentDistribution,
  lastUpdated,
  onDelete,
}: ProductCardProps) {
  const positivePercentage = Math.round((sentimentDistribution.positive / sentimentDistribution.total) * 100)

  const handleDelete = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (onDelete) {
      onDelete(name)
    }
  }

  return (
    <Card className="hover:shadow-lg transition-all duration-200 hover:border-primary/50 group h-full relative">
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <Link href={`/products/${encodeURIComponent(name)}`} className="flex-1 min-w-0">
            <CardTitle className="line-clamp-2 group-hover:text-primary transition-colors">{name}</CardTitle>
          </Link>
          {onDelete && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-muted-foreground hover:text-destructive shrink-0"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete Product</AlertDialogTitle>
                  <AlertDialogDescription>
                    Are you sure you want to delete "{name}"? This will permanently delete all {totalReviews} reviews for this product. This action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={handleDelete}
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  >
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
        </div>
      </CardHeader>
      <Link href={`/products/${encodeURIComponent(name)}`} className="block">
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
      </Link>
    </Card>
  )
}
