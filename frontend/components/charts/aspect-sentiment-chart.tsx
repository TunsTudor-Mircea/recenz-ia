"use client"

import { useState, useEffect } from "react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { AspectPolarityDistribution } from "@/types/api"

export const ASPECT_LABELS: Record<string, string> = {
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

interface AspectSentimentChartProps {
  data: AspectPolarityDistribution[]
}

export function AspectSentimentChart({ data }: AspectSentimentChartProps) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])

  const chartData = data
    .filter((d) => d.positive + d.negative + d.neutral > 0)
    .sort((a, b) => (b.positive + b.negative + b.neutral) - (a.positive + a.negative + a.neutral))
    .map((d) => ({
      aspect: ASPECT_LABELS[d.aspect] ?? d.aspect,
      Positive: d.positive,
      Negative: d.negative,
      Neutral: d.neutral,
    }))

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>Aspect Sentiment Breakdown</CardTitle>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <div>
          {!mounted ? (
            <div style={{ height: 320, minWidth: 860 }} />
          ) : (
            <BarChart width={860} height={320} data={chartData} layout="vertical" margin={{ left: 16, right: 24 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" opacity={0.5} horizontal={false} />
              <XAxis type="number" allowDecimals={false} stroke="#6b7280" tick={{ fill: "#374151" }} />
              <YAxis type="category" dataKey="aspect" width={90} stroke="#6b7280" tick={{ fill: "#374151", fontSize: 12 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#fff", border: "1px solid #e5e7eb", borderRadius: "8px" }}
                labelStyle={{ color: "#111827", fontWeight: 600 }}
              />
              <Legend wrapperStyle={{ color: "#111827" }} />
              <Bar dataKey="Positive" stackId="a" fill="#14b8a6" radius={[0, 0, 0, 0]} />
              <Bar dataKey="Negative" stackId="a" fill="#475569" radius={[0, 0, 0, 0]} />
              <Bar dataKey="Neutral" stackId="a" fill="#f59e0b" radius={[0, 4, 4, 0]} />
            </BarChart>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
