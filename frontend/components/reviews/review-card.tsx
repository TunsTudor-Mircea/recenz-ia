import { Card, CardContent } from "@/components/ui/card"
import { StarRating } from "@/components/star-rating"
import { SentimentBadge } from "@/components/sentiment-badge"
import { Calendar } from "lucide-react"
import type { Review } from "@/types/api"

interface ReviewCardProps {
  review: Review
}

export function ReviewCard({ review }: ReviewCardProps) {
  return (
    <Card>
      <CardContent className="pt-6 space-y-3">
        <div className="flex items-start justify-between gap-4">
          <StarRating rating={review.rating} size={16} showNumber={false} />
          <SentimentBadge sentiment={review.sentiment_label} score={review.sentiment_score} />
        </div>

        <p className="text-sm leading-relaxed">{review.review_text}</p>

        <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
          <div className="flex items-center gap-1">
            <Calendar className="h-3 w-3" />
            <span>{new Date(review.review_date).toLocaleDateString()}</span>
          </div>
          <span className="text-[10px] uppercase tracking-wider bg-muted px-2 py-1 rounded">{review.model_used}</span>
        </div>
      </CardContent>
    </Card>
  )
}
