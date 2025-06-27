import React from 'react';
import { CheckCircle, XCircle, Info, Database, Layers, Calendar } from 'lucide-react';
import type { HealthResponse, ModelInfo } from '../types';
import { formatNumber, formatDate } from '../utils/formatters';

interface ModelStatusProps {
  health: HealthResponse | null;
  modelInfo: ModelInfo | null;
  isLoading: boolean;
}

const ModelStatus: React.FC<ModelStatusProps> = ({ health, modelInfo, isLoading }) => {
  if (isLoading) {
    return (
      <div className="bg-gray-100 rounded-lg p-4">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-300 rounded w-1/3 mb-2"></div>
          <div className="h-3 bg-gray-300 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  if (!health) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center space-x-2">
          <XCircle className="w-5 h-5 text-red-600" />
          <span className="text-red-800 font-medium">Unable to connect to model</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center space-x-2 mb-4">
        <Info className="w-5 h-5 text-blue-600" />
        <h3 className="text-lg font-semibold text-gray-800">Model Status</h3>
      </div>

      {/* Health Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="bg-green-50 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <CheckCircle className="w-4 h-4 text-green-600" />
            <span className="text-sm font-medium text-green-800">Status</span>
          </div>
          <div className="text-green-900 font-bold capitalize">{health.status}</div>
        </div>

        <div className="bg-blue-50 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <Database className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-800">Features</span>
          </div>
          <div className="text-blue-900 font-bold">{health.features_count}</div>
        </div>

        <div className="bg-purple-50 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <Layers className="w-4 h-4 text-purple-600" />
            <span className="text-sm font-medium text-purple-800">Sequence</span>
          </div>
          <div className="text-purple-900 font-bold">{health.sequence_length} days</div>
        </div>

        {modelInfo && (
          <div className="bg-orange-50 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-1">
              <Calendar className="w-4 h-4 text-orange-600" />
              <span className="text-sm font-medium text-orange-800">Epochs</span>
            </div>
            <div className="text-orange-900 font-bold">{modelInfo.metadata.epochs}</div>
          </div>
        )}
      </div>

      {/* Model Details */}
      {modelInfo && (
        <div className="border-t border-gray-200 pt-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-3">Model Details</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Architecture:</span>
              <span className="ml-2 font-medium">LSTM with Attention</span>
            </div>
            <div>
              <span className="text-gray-600">Hidden Size:</span>
              <span className="ml-2 font-medium">{modelInfo.model_config.hidden_size}</span>
            </div>
            <div>
              <span className="text-gray-600">Layers:</span>
              <span className="ml-2 font-medium">{modelInfo.model_config.num_layers}</span>
            </div>
            <div>
              <span className="text-gray-600">Training Sequences:</span>
              <span className="ml-2 font-medium">{formatNumber(modelInfo.metadata.total_sequences, 0)}</span>
            </div>
            <div>
              <span className="text-gray-600">Final Loss:</span>
              <span className="ml-2 font-medium">{formatNumber(modelInfo.metadata.final_loss, 6)}</span>
            </div>
            <div>
              <span className="text-gray-600">Trained:</span>
              <span className="ml-2 font-medium">{formatDate(modelInfo.metadata.training_date)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelStatus;
