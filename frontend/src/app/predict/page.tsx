"use client";

import { useEffect, useState } from "react";
import { getMetadata, predictCrop, submitFeedback } from "@/lib/api";
import { MetadataResponse, PredictionRequest, PredictionResponse } from "@/types";

export default function PredictPage() {
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [feedbackSent, setFeedbackSent] = useState(false);

  useEffect(() => {
    async function fetchMeta() {
      try {
        const data = await getMetadata();
        setMetadata(data);
        
        // Initialize default form data
        const initialData: Record<string, any> = {};
        data.features.forEach(f => {
          if (f.kind === "category" && f.choices?.length) {
            initialData[f.name] = f.choices[0];
          } else if (f.kind === "number") {
            // Set some midpoint or minimum as default
            initialData[f.name] = f.minimum || 0;
          }
        });
        setFormData(initialData);
      } catch (err: any) {
        setError(err.message || "Failed to load model metadata.");
      } finally {
        setLoading(false);
      }
    }
    fetchMeta();
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === "number" ? Number(value) : value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setPredicting(true);
    setError(null);
    setResult(null);
    setFeedbackSent(false);

    try {
      // Create request payload matching API exactly
      const payload = { ...formData } as PredictionRequest;
      const res = await predictCrop(payload);
      setResult(res);
    } catch (err: any) {
      setError(err.message || "Prediction failed.");
    } finally {
      setPredicting(false);
    }
  };

  const handleFeedback = async (helpful: boolean) => {
    if (!result) return;
    try {
      await submitFeedback(result.prediction_id, { helpful });
      setFeedbackSent(true);
    } catch (err: any) {
      alert("Failed to submit feedback.");
    }
  };

  if (loading) {
    return <div className="p-8 text-center text-slate-500">Loading form...</div>;
  }

  if (error && !metadata) {
    return (
      <div className="bg-red-50 text-red-700 p-4 rounded-md border border-red-200">
        <h2 className="font-bold">Error</h2>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <header>
        <h1 className="text-3xl font-bold text-slate-800">Crop Recommendation</h1>
        <p className="text-slate-600 mt-2">Enter agricultural parameters to get a federated learning-based crop recommendation.</p>
      </header>

      {error && (
        <div className="bg-red-50 text-red-700 p-4 rounded-md border border-red-200">
          <p>{error}</p>
        </div>
      )}

      <div className="grid md:grid-cols-2 gap-8 items-start">
        <form onSubmit={handleSubmit} className="bg-white p-6 rounded-lg shadow-sm border border-slate-200 space-y-6">
          {metadata?.features.map((feature) => (
            <div key={feature.name}>
              <label className="block text-sm font-medium text-slate-700 mb-1 capitalize">
                {feature.name.replace(/_/g, ' ')}
              </label>
              {feature.kind === "category" ? (
                <select
                  name={feature.name}
                  value={formData[feature.name] || ""}
                  onChange={handleChange}
                  className="w-full border-slate-300 rounded-md shadow-sm focus:border-emerald-500 focus:ring-emerald-500 p-2 border"
                  required
                >
                  {feature.choices?.map(choice => (
                    <option key={choice} value={choice}>{choice}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  name={feature.name}
                  value={formData[feature.name] || ""}
                  onChange={handleChange}
                  min={feature.minimum}
                  max={feature.maximum}
                  step="0.1"
                  className="w-full border-slate-300 rounded-md shadow-sm focus:border-emerald-500 focus:ring-emerald-500 p-2 border"
                  required
                />
              )}
              {feature.kind === "number" && (
                <p className="text-xs text-slate-500 mt-1">
                  Range: {feature.minimum} - {feature.maximum}
                </p>
              )}
            </div>
          ))}

          <button
            type="submit"
            disabled={predicting}
            className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-medium py-3 px-4 rounded-md shadow transition-colors disabled:opacity-50"
          >
            {predicting ? "Predicting..." : "Predict Crop"}
          </button>
        </form>

        {result && (
          <div className="bg-emerald-50 p-6 rounded-lg shadow-sm border border-emerald-200 sticky top-6">
            <h2 className="text-xl font-bold text-emerald-800 mb-4">Recommended Crop</h2>
            
            <div className="bg-white p-6 rounded-md shadow-inner text-center mb-6">
              <div className="text-4xl font-bold text-emerald-700 capitalize">
                {result.predicted_crop}
              </div>
              <div className="text-sm text-slate-500 mt-2">
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </div>
            </div>

            {result.confidence < 0.5 && (
              <div className="mb-6 bg-amber-50 border border-amber-200 text-amber-800 p-3 rounded-md text-sm">
                <span className="font-semibold">⚠️ Low confidence prediction</span> — additional field information is recommended.
              </div>
            )}

            {result.recommendations && result.recommendations.length > 1 && (
              <div className="mb-6">
                <h3 className="text-sm font-semibold text-slate-700 mb-2">Alternative Recommendations</h3>
                <ul className="space-y-2 text-sm text-slate-600">
                  {result.recommendations.slice(1).map((rec) => (
                    <li key={rec.rank} className="flex justify-between bg-white px-3 py-2 rounded border border-emerald-100">
                      <span className="capitalize">{rec.rank}. {rec.crop}</span>
                      <span>{(rec.confidence * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="text-xs text-emerald-700/70 mb-6 space-y-1">
              <p>Model: {metadata?.name}</p>
              <p>Version: {result.model_version}</p>
              <p>Request ID: {result.prediction_id}</p>
            </div>

            <div className="border-t border-emerald-200 pt-4">
              {feedbackSent ? (
                <p className="text-sm text-emerald-700 font-medium text-center">Thank you for your feedback!</p>
              ) : (
                <div>
                  <p className="text-sm text-slate-700 text-center mb-3">Was this recommendation useful?</p>
                  <div className="flex space-x-4 justify-center">
                    <button
                      onClick={() => handleFeedback(true)}
                      className="px-4 py-2 bg-white border border-emerald-300 text-emerald-700 rounded hover:bg-emerald-100 transition-colors"
                    >
                      👍 Correct
                    </button>
                    <button
                      onClick={() => handleFeedback(false)}
                      className="px-4 py-2 bg-white border border-red-300 text-red-700 rounded hover:bg-red-50 transition-colors"
                    >
                      👎 Incorrect
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
