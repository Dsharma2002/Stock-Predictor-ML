import { useState, useEffect } from 'react';
import { Brain, TrendingUp } from 'lucide-react';
import TickerInput from './components/TickerInput';
import PredictionResult from './components/PredictionResult';
import ModelStatus from './components/ModelStatus';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';
import StockAnalysis from './components/StockAnalysis';
import StockPredictionAPI from './services/api';
import type { PredictionResponse, HealthResponse, ModelInfo } from './types';

function App() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [supportedTickers, setSupportedTickers] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load initial data on component mount
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        setIsInitialLoading(true);
        const [healthData, modelData, tickersData] = await Promise.all([
          StockPredictionAPI.getHealth(),
          StockPredictionAPI.getModelInfo(),
          StockPredictionAPI.getSupportedTickers(),
        ]);
        
        setHealth(healthData);
        setModelInfo(modelData);
        setSupportedTickers(tickersData.tickers);
        setError(null);
      } catch (err) {
        console.error('Failed to load initial data:', err);
        setError('Failed to connect to the prediction service. Please ensure the backend is running.');
      } finally {
        setIsInitialLoading(false);
      }
    };

    loadInitialData();
  }, []);

  const handlePredict = async (ticker: string) => {
    try {
      setIsLoading(true);
      setError(null);
      setPrediction(null);
      
      const result = await StockPredictionAPI.predictStock({ ticker });
      setPrediction(result);
    } catch (err) {
      console.error('Prediction failed:', err);
      setError(err instanceof Error ? err.message : 'Prediction failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  if (isInitialLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="mb-6">
            <Brain className="w-16 h-16 text-blue-600 mx-auto animate-pulse" />
          </div>
          <LoadingSpinner size="lg" text="Initializing AI Stock Predictor..." />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-600 rounded-lg p-2">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">AI Stock Predictor</h1>
              <p className="text-gray-600">LSTM-powered stock price predictions</p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Message */}
        {error && (
          <ErrorMessage 
            error={error}
            onRetry={() => {
              setError(null);
              // If there was a prediction error, we could retry the last ticker
            }}
            dismissible={true}
            onDismiss={() => setError(null)}
          />
        )}

        {/* Model Status */}
        <div className="mb-8">
          <ModelStatus 
            health={health} 
            modelInfo={modelInfo} 
            isLoading={isInitialLoading} 
          />
        </div>

        {/* Main Content */}
        <div className="space-y-8">
          {/* Input Section */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <div className="flex items-center space-x-2 mb-6">
              <TrendingUp className="w-6 h-6 text-blue-600" />
              <h2 className="text-xl font-semibold text-gray-800">Make Prediction</h2>
            </div>
            
            <TickerInput 
              onPredict={handlePredict}
              isLoading={isLoading}
              supportedTickers={supportedTickers}
            />

            {isLoading && (
              <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                <LoadingSpinner text="Analyzing stock data..." />
              </div>
            )}
          </div>

          {/* Prediction Result */}
          {prediction && (
            <PredictionResult prediction={prediction} />
          )}

          {/* Stock Analysis Section */}
          <StockAnalysis 
            ticker={prediction?.ticker || ''}
            prediction={prediction || undefined}
          />
        </div>

        {/* Footer */}
        <footer className="mt-12 text-center text-gray-500 text-sm">
          <p>
            Powered by PyTorch LSTM with Attention Mechanism | 
            Built with React, TypeScript & Tailwind CSS
          </p>
          <p className="mt-1">
            This tool is for educational purposes only and should not be used for actual financial decisions.
          </p>
        </footer>
      </main>
    </div>
  );
}

export default App;
