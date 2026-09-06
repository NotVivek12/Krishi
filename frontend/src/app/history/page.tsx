"use client";

import { useEffect, useState } from "react";
import { getPredictionHistory } from "@/lib/api";
import { PredictionHistoryItem } from "@/types";

export default function HistoryPage() {
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchHistory() {
      try {
        const data = await getPredictionHistory();
        setHistory(data);
      } catch (err: any) {
        setError(err.message || "Failed to load prediction history.");
      } finally {
        setLoading(false);
      }
    }
    fetchHistory();
  }, []);

  if (loading) {
    return <div className="p-8 text-center text-slate-500">Loading history...</div>;
  }

  if (error) {
    return (
      <div className="bg-red-50 text-red-700 p-4 rounded-md border border-red-200">
        <h2 className="font-bold">Error</h2>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-3xl font-bold text-slate-800">Prediction History</h1>
        <p className="text-slate-600 mt-2">View your recent crop recommendations.</p>
      </header>

      {history.length === 0 ? (
        <div className="bg-white p-8 rounded-lg shadow-sm border border-slate-200 text-center text-slate-500">
          No prediction history found. Make a prediction first!
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow-sm border border-slate-200 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-600">
              <thead className="bg-slate-50 text-slate-700 font-medium border-b border-slate-200">
                <tr>
                  <th className="px-6 py-4">Date</th>
                  <th className="px-6 py-4">Predicted Crop</th>
                  <th className="px-6 py-4">Confidence</th>
                  <th className="px-6 py-4">Model Version</th>
                  <th className="px-6 py-4">Feedback</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {history.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      {new Date(item.created_at).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 font-medium text-emerald-700 capitalize">
                      {item.predicted_crop}
                    </td>
                    <td className="px-6 py-4">
                      {(item.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 text-xs">
                      {item.model_version}
                    </td>
                    <td className="px-6 py-4">
                      {item.feedback ? (
                        item.feedback.helpful ? "👍 Correct" : "👎 Incorrect"
                      ) : (
                        <span className="text-slate-400">None</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
