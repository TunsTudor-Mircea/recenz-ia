"use client"

import { Card, CardContent } from "@/components/ui/card"
import { StarRating } from "@/components/star-rating"
import { SentimentBadge } from "@/components/sentiment-badge"
import { Calendar } from "lucide-react"
import type { Review } from "@/types/api"

const MODEL_LABELS: Record<string, string> = {
  absa_xlmr: "XLM-RoBERTa · ABSA",
  absa_robert: "RoBERT · ABSA",
  absa_mbert: "mBERT · ABSA",
  absa_lr: "LR · ABSA",
  absa_svm: "SVM · ABSA",
  robert: "RoBERT",
  xgboost: "XGBoost",
  svm: "SVM",
  lr: "Logistic Regression",
}

const ASPECT_LABELS: Record<string, string> = {
  BATERIE: "Battery",
  ECRAN: "Screen",
  SUNET: "Sound",
  PERFORMANTA: "Performance",
  CONECTIVITATE: "Connectivity",
  DESIGN: "Design",
  CALITATE_CONSTRUCTIE: "Build Quality",
  PRET: "Price",
  LIVRARE: "Delivery",
  GENERAL: "General",
}

const POLARITY_STYLES: Record<string, string> = {
  positive: "bg-green-100 text-green-700",
  negative: "bg-red-100 text-red-700",
  neutral: "bg-yellow-100 text-yellow-700",
}

interface ReviewCardProps {
  review: Review
}

export function ReviewCard({ review }: ReviewCardProps) {
  const mentionedAspects = review.aspects
    ? Object.entries(review.aspects).filter(([, polarity]) => polarity !== "none")
    : []

  return (
    <Card>
      <CardContent className="pt-4 pb-4 space-y-2">
        <div className="flex items-center gap-3 flex-wrap">
          <StarRating rating={review.rating} size={16} showNumber={false} />
          <SentimentBadge sentiment={review.sentiment_label} score={review.sentiment_score} />
        </div>

        {review.review_title && (
          <h3 className="font-semibold text-base leading-tight">{review.review_title}</h3>
        )}

        <p className="text-sm leading-relaxed text-muted-foreground">{review.review_text}</p>

        {mentionedAspects.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {mentionedAspects.map(([aspect, polarity]) => (
              <span
                key={aspect}
                className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${POLARITY_STYLES[polarity] ?? "bg-gray-100 text-gray-600"}`}
              >
                {ASPECT_LABELS[aspect] ?? aspect}
              </span>
            ))}
          </div>
        )}

        <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
          <div className="flex items-center gap-1">
            <Calendar className="h-3 w-3" />
            <span>{new Date(review.review_date).toLocaleDateString()}</span>
          </div>
          <span className="text-[10px] uppercase tracking-wider bg-muted px-2 py-1 rounded">
            {MODEL_LABELS[review.model_used] ?? review.model_used}
          </span>
        </div>
      </CardContent>
    </Card>
  )
}
