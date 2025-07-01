import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, BarChart3 } from 'lucide-react';
import StockChart from './StockChart';
import StockNews from './StockNews';
import LoadingSpinner from './LoadingSpinner';
import StockPredictionAPI from '../services/api';
import type { StockDataResponse, NewsResponse, PredictionResponse } from '../types';

interface StockAnalysisProps {
  ticker: string;
  prediction?: PredictionResponse;
}

type TimePeriod = '1d' | '1w' | '1m' | '1y';

const StockAnalysis: React.FC<StockAnalysisProps> = ({ ticker, prediction }) => {
  const [stockData, setStockData] = useState<StockDataResponse | null>(null);
  const [news, setNews] = useState<NewsResponse | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<TimePeriod>('1w');
  const [isLoadingChart, setIsLoadingChart] = useState(false);
  const [isLoadingNews, setIsLoadingNews] = useState(false);
  const [chartError, setChartError] = useState<string | null>(null);

  const timePeriods = [
    { key: '1d' as TimePeriod, label: '1D', description: '1 Day' },
    { key: '1w' as TimePeriod, label: '1W', description: '1 Week' },
    { key: '1m' as TimePeriod, label: '1M', description: '1 Month' },
    { key: '1y' as TimePeriod, label: '1Y', description: '1 Year' },
  ];

  useEffect(() => {
    if (ticker) {
      loadStockData(selectedPeriod);
      loadNews();
    }
  }, [ticker, selectedPeriod]);

  const loadStockData = async (period: TimePeriod) => {
    try {
      setIsLoadingChart(true);
      setChartError(null);
      const data = await StockPredictionAPI.getStockData({ ticker, period });
      setStockData(data);
    } catch (error) {
      console.error('Error loading stock data:', error);
      setChartError(error instanceof Error ? error.message : 'Failed to load chart data');
    } finally {
      setIsLoadingChart(false);
    }
  };

  const loadNews = async () => {
    try {
      setIsLoadingNews(true);
      const newsData = await StockPredictionAPI.getStockNews(ticker);
      setNews(newsData);
    } catch (error) {
      console.error('Error loading news:', error);
      // Don't show error for news, just set empty news
      setNews({ ticker, news: [] });
    } finally {
      setIsLoadingNews(false);
    }
  };

  const handlePeriodChange = (period: TimePeriod) => {
    setSelectedPeriod(period);
  };

  if (!ticker) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-12 border border-gray-200 text-center">
        <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-gray-600 mb-2">
          Stock Analysis
        </h3>
        <p className="text-gray-500">
          Select a stock ticker to view detailed charts and analysis
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stock Chart Section */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <BarChart3 className="w-6 h-6 text-blue-600" />
              <div>
                <h3 className="text-xl font-semibold text-gray-800">{ticker} Stock Chart</h3>
                {stockData && (
                  <div className="flex items-center space-x-4 mt-1">
                    <span className="text-2xl font-bold text-gray-900">
                      ${stockData.current_price}
                    </span>
                    <div className={`flex items-center space-x-1 ${
                      stockData.price_change >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {stockData.price_change >= 0 ? (
                        <TrendingUp className="w-4 h-4" />
                      ) : (
                        <TrendingDown className="w-4 h-4" />
                      )}
                      <span className="font-medium">
                        {stockData.price_change >= 0 ? '+' : ''}
                        {stockData.price_change} ({stockData.price_change_percent >= 0 ? '+' : ''}
                        {stockData.price_change_percent.toFixed(2)}%)
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            {/* Time Period Buttons */}
            <div className="flex items-center space-x-1 bg-white rounded-lg p-1 border border-gray-200">
              {timePeriods.map((period) => (
                <button
                  key={period.key}
                  onClick={() => handlePeriodChange(period.key)}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    selectedPeriod === period.key
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                  title={period.description}
                >
                  {period.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Chart Content */}
        <div className="p-6">
          {isLoadingChart ? (
            <div className="flex items-center justify-center h-96">
              <LoadingSpinner text="Loading chart data..." />
            </div>
          ) : chartError ? (
            <div className="flex items-center justify-center h-96">
              <div className="text-center">
                <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500 mb-4">{chartError}</p>
                <button
                  onClick={() => loadStockData(selectedPeriod)}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                >
                  Retry
                </button>
              </div>
            </div>
          ) : stockData && stockData.data.length > 0 ? (
            <div>
              <StockChart data={stockData.data} height={400} />
              <div className="mt-4 text-sm text-gray-500 text-center">
                Showing {stockData.period} data for {stockData.ticker} • {stockData.data.length} data points
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-96">
              <div className="text-center">
                <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No chart data available for {ticker}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Prediction Summary */}
      {prediction && (
        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
          <h4 className="text-lg font-semibold text-gray-800 mb-4">AI Prediction Summary</h4>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Current Price</p>
              <p className="text-xl font-bold text-gray-900">${prediction.current_price}</p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Predicted Price</p>
              <p className="text-xl font-bold text-gray-900">${prediction.predicted_price}</p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Change</p>
              <p className={`text-xl font-bold ${prediction.prediction_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {prediction.prediction_change >= 0 ? '+' : ''}${prediction.prediction_change}
              </p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Confidence</p>
              <p className={`text-xl font-bold ${
                prediction.confidence === 'high' ? 'text-green-600' : 
                prediction.confidence === 'medium' ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {prediction.confidence.toUpperCase()}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* News Section */}
      <StockNews
        news={news?.news || []}
        ticker={ticker}
        isLoading={isLoadingNews}
      />
    </div>
  );
};

export default StockAnalysis;
