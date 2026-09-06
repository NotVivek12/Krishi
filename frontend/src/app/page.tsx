"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getHealth, getMetadata } from "@/lib/api";
import { HealthResponse, MetadataResponse } from "@/types";

export default function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [healthData, metaData] = await Promise.all([
          getHealth().catch((e) => {
            console.error(e);
            return { status: "offline", checks: {} } as HealthResponse;
          }),
          getMetadata(),
        ]);
        setHealth(healthData);
        setMetadata(metaData);
      } catch (err: any) {
        setError(err.message || "Failed to load backend data.");
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  return (
    <div className="space-y-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-slate-800">AgriFL Dashboard</h1>
        <p className="text-slate-600 mt-2">Privacy-Preserving Intelligent Crop Recommendation using Federated Learning</p>
      </header>

      {loading ? (
        <div className="animate-pulse flex space-x-4">
          <div className="flex-1 space-y-4 py-1">
            <div className="h-4 bg-slate-200 rounded w-3/4"></div>
            <div className="h-4 bg-slate-200 rounded"></div>
            <div className="h-4 bg-slate-200 rounded w-5/6"></div>
          </div>
        </div>
      ) : error ? (
        <div className="bg-red-50 text-red-700 p-4 rounded-md border border-red-200">
          <h2 className="font-bold">Error Connecting to Backend</h2>
          <p>{error}</p>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
            <h2 className="text-lg font-semibold text-slate-800 mb-4 border-b pb-2">Backend Status</h2>
            <div className="flex items-center space-x-3 text-lg">
              <span
                className={`w-3 h-3 rounded-full ${
                  health?.status === "healthy" ? "bg-emerald-500" : "bg-red-500"
                }`}
              ></span>
              <span className="font-medium text-slate-700">
                {health?.status === "healthy" ? "Online" : "Offline"}
              </span>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
            <h2 className="text-lg font-semibold text-slate-800 mb-4 border-b pb-2">System Information</h2>
            <dl className="space-y-3 text-sm">
              <div className="flex justify-between">
                <dt className="text-slate-500">Active Model</dt>
                <dd className="font-medium text-emerald-700">
                  {metadata?.name || "Unknown"}
                  {metadata?.version && ` (${metadata.version})`}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">Supported Crops</dt>
                <dd className="font-medium text-slate-800">{metadata?.output_classes || 0} classes</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-slate-500">Input Features</dt>
                <dd className="font-medium text-slate-800">{metadata?.features?.length || 0} features</dd>
              </div>
              <div className="flex flex-col mt-4 pt-4 border-t">
                <dt className="text-slate-500 mb-1">Model Limitations</dt>
                <dd className="text-xs text-slate-600">{metadata?.limitations}</dd>
              </div>
            </dl>
          </div>
        </div>
      )}

      <div className="flex justify-center mt-12">
        <Link 
          href="/predict" 
          className="bg-emerald-600 hover:bg-emerald-700 text-white font-medium py-3 px-8 rounded-md shadow transition-colors text-lg"
        >
          Get Crop Recommendation
        </Link>
      </div>
    </div>
  );
}
