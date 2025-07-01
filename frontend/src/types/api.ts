export interface PredictionRequest {
  ticker: string;
  days_ahead?: number;
}

export interface PredictionResponse {
  ticker: string;
  current_price: number;
  predicted_price: number;
  prediction_change: number;
  prediction_change_percent: number;
  confidence: 'high' | 'medium' | 'low';
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  features_count: number;
  sequence_length: number;
}

export interface ModelInfo {
  model_config: {
    input_size: number;
    hidden_size: number;
    num_layers: number;
    dropout: number;
  };
  feature_columns: string[];
  sequence_length: number;
  metadata: {
    training_date: string;
    tickers: string[];
    total_sequences: number;
    epochs: number;
    final_loss: number;
    feature_count: number;
  };
}

export interface StockDataRequest {
  ticker: string;
  period: '1d' | '1w' | '1m' | '1y';
}

export interface CandlestickData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StockDataResponse {
  ticker: string;
  period: string;
  data: CandlestickData[];
  current_price: number;
  price_change: number;
  price_change_percent: number;
}

export interface NewsItem {
  title: string;
  url: string;
  source: string;
  published_at: string;
  summary?: string;
}

export interface NewsResponse {
  ticker: string;
  news: NewsItem[];
}

export interface SupportedTickers {
  tickers: string[];
  note: string;
}

export interface ApiError {
  detail: string;
}
