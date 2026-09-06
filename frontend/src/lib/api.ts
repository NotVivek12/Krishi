import {
  HealthResponse,
  MetadataResponse,
  PredictionRequest,
  PredictionResponse,
  PredictionHistoryItem,
  FeedbackRequest,
  FeedbackResponse,
} from "../types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_URL}${endpoint}`;
  
  const headers = {
    "Content-Type": "application/json",
    ...(options?.headers || {}),
  };

  const response = await fetch(url, { ...options, headers });
  
  if (!response.ok) {
    let errorMessage = "An unknown error occurred.";
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || JSON.stringify(errorData);
    } catch {
      errorMessage = response.statusText;
    }
    throw new ApiError(response.status, errorMessage);
  }

  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return fetchApi<HealthResponse>("/health/");
}

export async function getMetadata(): Promise<MetadataResponse> {
  return fetchApi<MetadataResponse>("/model/metadata/");
}

export async function predictCrop(payload: PredictionRequest): Promise<PredictionResponse> {
  return fetchApi<PredictionResponse>("/predict/", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// In a real application, we would pass auth tokens here if using authentication.
// For this frontend without auth configured, the backend might reject this, 
// so we need to see how the backend handles anonymous history/feedback, or if we mock a user.
// The backend requires IsAuthenticated for /predictions/ and /feedback/, which means we might need a token.
// Let's implement the API calls, but we may need to bypass auth if it's a simple project frontend,
// or just send requests and handle 403s.
export async function getPredictionHistory(): Promise<PredictionHistoryItem[]> {
  return fetchApi<PredictionHistoryItem[]>("/predictions/");
}

export async function submitFeedback(predictionId: string, payload: FeedbackRequest): Promise<FeedbackResponse> {
  return fetchApi<FeedbackResponse>(`/predictions/${predictionId}/feedback/`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
