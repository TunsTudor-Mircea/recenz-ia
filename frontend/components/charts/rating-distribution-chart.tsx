"use client"

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { RatingDistribution } from "@/types/api"

interface RatingDistributionChartProps {
  data: RatingDistribution
}

// Medium vibrancy teal - balanced color
const CHART_COLOR = "#14b8a6" // Medium teal

export function RatingDistributionChart({ data }: RatingDistributionChartProps) {
  const chartData = [
    { rating: "5 ★", count: data.rating_5 },
    { rating: "4 ★", count: data.rating_4 },
    { rating: "3 ★", count: data.rating_3 },
    { rating: "2 ★", count: data.rating_2 },
    { rating: "1 ★", count: data.rating_1 },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle>Rating Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" opacity={0.5} />
            <XAxis type="number" stroke="#6b7280" tick={{ fill: "#6b7280" }} />
            <YAxis type="category" dataKey="rating" stroke="#6b7280" tick={{ fill: "#374151" }} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#ffffff",
                border: "1px solid #e5e7eb",
                borderRadius: "8px"
              }}
              labelStyle={{ color: "#111827" }}
            />
            <Bar dataKey="count" radius={[0, 4, 4, 0]}>
              {chartData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={CHART_COLOR} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
