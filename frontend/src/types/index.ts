export interface HealthResponse {
  status: string;
  checks: {
    database?: { status: string };
    model?: { status: string; model_version?: string; detail?: string };
  };
}

export interface FeatureMetadata {
  name: string;
  model_name: string;
  kind: "category" | "number";
  choices?: string[];
  minimum?: number;
  maximum?: number;
}

export interface MetadataResponse {
  name: string;
  version: string;
  status: string;
  output_classes: number;
  default_top_k: number;
  maximum_top_k: number;
  features: FeatureMetadata[];
  limitations: string;
}

export interface PredictionRequest {
  soil: string;
  season: string;
  sown: string;
  water_source: string;
  soil_ph: number;
  crop_duration: number;
  temperature: number;
  water_required: number;
  relative_humidity: number;
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  top_k?: number;
}

export interface Recommendation {
  rank: number;
  crop: string;
  confidence: number;
}

export interface PredictionResponse {
  prediction_id: string;
  predicted_crop: string;
  confidence: number;
  recommendations: Recommendation[];
  model_version: string;
  created_at: string;
}

export interface FeedbackRequest {
  helpful: boolean;
  actual_crop?: string;
  notes?: string;
}

export interface FeedbackResponse {
  helpful: boolean;
  actual_crop: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export interface PredictionHistoryItem {
  id: string;
  input_data: PredictionRequest;
  predicted_crop: string;
  confidence: number;
  recommendations: Recommendation[];
  model_version: string;
  feedback: FeedbackResponse | null;
  created_at: string;
}
