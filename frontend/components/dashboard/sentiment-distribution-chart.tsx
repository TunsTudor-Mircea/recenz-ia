"use client"

import { Cell, Pie, PieChart, ResponsiveContainer, Legend, Tooltip } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { SentimentDistribution } from "@/types/api"

interface SentimentDistributionChartProps {
  data: SentimentDistribution
}

// Professional colors matching app theme - neutral palette
const COLORS = {
  positive: "#0891b2", // Cyan-600 - bright teal for positive
  neutral: "#64748b", // Slate-500 - neutral gray-blue
  negative: "#475569", // Slate-600 - darker neutral gray
}

export function SentimentDistributionChart({ data }: SentimentDistributionChartProps) {
  const chartData = [
    { name: "Positive", value: data.positive, color: COLORS.positive },
    { name: "Neutral", value: data.neutral, color: COLORS.neutral },
    { name: "Negative", value: data.negative, color: COLORS.negative },
  ]

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sentiment Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie data={chartData} cx="50%" cy="50%" labelLine={false} outerRadius={100} fill="#8884d8" dataKey="value">
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
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
          </PieChart>
        </ResponsiveContainer>
        <div className="mt-4 grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold" style={{ color: COLORS.positive }}>
              {data.positive}
            </div>
            <div className="text-xs text-muted-foreground">Positive</div>
          </div>
          <div>
            <div className="text-2xl font-bold" style={{ color: COLORS.neutral }}>
              {data.neutral}
            </div>
            <div className="text-xs text-muted-foreground">Neutral</div>
          </div>
          <div>
            <div className="text-2xl font-bold" style={{ color: COLORS.negative }}>
              {data.negative}
            </div>
            <div className="text-xs text-muted-foreground">Negative</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
