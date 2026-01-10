export interface Review {
  id: string
  product_name: string
  review_title?: string | null
  review_text: string
  rating: 1 | 2 | 3 | 4 | 5
  sentiment_label: "positive" | "negative"
  sentiment_score: number
  model_used: "robert" | "xgboost" | "svm"
  review_date: string
  created_at: string
}

export interface SentimentDistribution {
  positive: number
  negative: number
  total: number
}

export interface RatingDistribution {
  rating_1: number
  rating_2: number
  rating_3: number
  rating_4: number
  rating_5: number
  average_rating: number
  total: number
}

export interface AnalyticsSummary {
  total_reviews: number
  total_products: number
  average_rating: number
  sentiment_distribution: SentimentDistribution
  top_rated_products: string[]
  most_reviewed_products: string[]
}

export interface ProductAnalytics {
  product_name: string
  total_reviews: number
  average_rating: number
  average_sentiment_score: number
  sentiment_distribution: SentimentDistribution
  rating_distribution: RatingDistribution
  oldest_review_date: string
  newest_review_date: string
}

export interface TrendDataPoint {
  date: string
  count: number
  average_rating: number
  average_sentiment: number
}

export interface ProductTrend {
  product_name: string
  period: string
  data_points: TrendDataPoint[]
}

export interface ScrapingJob {
  id: string
  url: string
  status: "pending" | "running" | "completed" | "failed"
  reviews_scraped: number
  reviews_created: number
  error_message: string | null
  job_metadata: Record<string, unknown>
  created_at: string
  completed_at: string | null
}

export interface PaginatedReviews {
  reviews: Review[]
  total: number
  page: number
  page_size: number
}

export interface PaginatedJobs {
  jobs: ScrapingJob[]
  total: number
  page: number
  page_size: number
}
