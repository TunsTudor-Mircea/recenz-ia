"use client"

import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ProductTrend } from "@/types/api"

interface SentimentTrendChartProps {
  data: ProductTrend
}

export function SentimentTrendChart({ data }: SentimentTrendChartProps) {
  const chartData = data.data_points.map((point) => ({
    date: new Date(point.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
    sentiment: point.average_sentiment.toFixed(2),
    rating: point.average_rating.toFixed(1),
    count: point.count,
  }))

  return (
    <Card className="lg:col-span-2">
      <CardHeader>
        <CardTitle>Sentiment Trend Over Time</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" opacity={0.5} />
            <XAxis
              dataKey="date"
              stroke="#6b7280"
              tick={{ fill: "#374151" }}
            />
            <YAxis
              yAxisId="left"
              domain={[0, 1]}
              stroke="#22c55e"
              tick={{ fill: "#6b7280" }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              domain={[0, 5]}
              stroke="#f59e0b"
              tick={{ fill: "#6b7280" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#ffffff",
                border: "1px solid #e5e7eb",
                borderRadius: "8px"
              }}
              labelStyle={{ color: "#111827" }}
            />
            <Legend
              wrapperStyle={{ color: "#111827" }}
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="sentiment"
              stroke="#22c55e"
              strokeWidth={3}
              name="Sentiment Score"
              dot={{ fill: "#22c55e", strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, fill: "#22c55e" }}
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="rating"
              stroke="#f59e0b"
              strokeWidth={3}
              name="Average Rating"
              dot={{ fill: "#f59e0b", strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, fill: "#f59e0b" }}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
