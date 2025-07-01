import axios from 'axios';
import type {
  PredictionRequest,
  PredictionResponse,
  HealthResponse,
  ModelInfo,
  SupportedTickers,
  StockDataRequest,
  StockDataResponse,
  NewsResponse,
} from '../types';

const API_BASE_URL = 'http://localhost/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw error;
  }
);

export class StockPredictionAPI {
  static async getHealth(): Promise<HealthResponse> {
    const response = await apiClient.get<HealthResponse>('/health');
    return response.data;
  }

  static async getModelInfo(): Promise<ModelInfo> {
    const response = await apiClient.get<ModelInfo>('/model-info');
    return response.data;
  }

  static async getSupportedTickers(): Promise<SupportedTickers> {
    const response = await apiClient.get<SupportedTickers>('/supported-tickers');
    return response.data;
  }

  static async predictStock(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await apiClient.post<PredictionResponse>('/predict', request);
    return response.data;
  }

  static async getStockData(request: StockDataRequest): Promise<StockDataResponse> {
    const response = await apiClient.post<StockDataResponse>('/stock-data', request);
    return response.data;
  }

  static async getStockNews(ticker: string): Promise<NewsResponse> {
    const response = await apiClient.get<NewsResponse>(`/news/${ticker}`);
    return response.data;
  }
}

export default StockPredictionAPI;
