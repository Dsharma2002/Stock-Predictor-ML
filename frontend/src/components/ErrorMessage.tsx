import React from 'react';
import { AlertTriangle, XCircle, Info, RefreshCw } from 'lucide-react';

interface ErrorMessageProps {
  error: string;
  onRetry?: () => void;
  type?: 'error' | 'warning' | 'info';
  dismissible?: boolean;
  onDismiss?: () => void;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ 
  error, 
  onRetry, 
  type = 'error',
  dismissible = false,
  onDismiss 
}) => {
  const getErrorDetails = (errorMessage: string) => {
    // Parse different types of errors and provide user-friendly messages
    if (errorMessage.includes('404') && errorMessage.includes('No data found for ticker')) {
      const ticker = errorMessage.match(/ticker (\w+)/)?.[1] || 'Unknown';
      return {
        title: 'Stock Not Found',
        message: `We couldn't find data for "${ticker}". This could mean:`,
        suggestions: [
          'The ticker symbol might be incorrect',
          'The stock might not be publicly traded',
          'Try searching for a different ticker symbol',
          'Check if the company is listed on major exchanges'
        ],
        icon: <Info className="w-5 h-5" />,
        bgColor: 'bg-blue-50',
        borderColor: 'border-blue-200',
        textColor: 'text-blue-800',
        iconColor: 'text-blue-600'
      };
    }
    
    if (errorMessage.includes('timeout') || errorMessage.includes('network')) {
      return {
        title: 'Connection Problem',
        message: 'Unable to connect to the prediction service.',
        suggestions: [
          'Check your internet connection',
          'The server might be temporarily unavailable',
          'Try again in a few moments'
        ],
        icon: <XCircle className="w-5 h-5" />,
        bgColor: 'bg-red-50',
        borderColor: 'border-red-200',
        textColor: 'text-red-800',
        iconColor: 'text-red-600'
      };
    }

    if (errorMessage.includes('Insufficient data')) {
      return {
        title: 'Insufficient Data',
        message: 'Not enough historical data available for this stock.',
        suggestions: [
          'This stock might be newly listed',
          'Try a more established company ticker',
          'Historical data might not be available'
        ],
        icon: <AlertTriangle className="w-5 h-5" />,
        bgColor: 'bg-yellow-50',
        borderColor: 'border-yellow-200',
        textColor: 'text-yellow-800',
        iconColor: 'text-yellow-600'
      };
    }

    // Default error handling
    return {
      title: 'Prediction Error',
      message: 'An error occurred while processing your request.',
      suggestions: [
        'Please try again with a different ticker',
        'Check that the ticker symbol is valid',
        'Contact support if the problem persists'
      ],
      icon: <AlertTriangle className="w-5 h-5" />,
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      textColor: 'text-red-800',
      iconColor: 'text-red-600'
    };
  };

  const errorDetails = getErrorDetails(error);

  return (
    <div className={`${errorDetails.bgColor} border ${errorDetails.borderColor} rounded-lg p-6 mb-6`}>
      <div className="flex items-start space-x-3">
        <div className={`${errorDetails.iconColor} mt-0.5 flex-shrink-0`}>
          {errorDetails.icon}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className={`text-lg font-semibold ${errorDetails.textColor} mb-2`}>
            {errorDetails.title}
          </h3>
          <p className={`${errorDetails.textColor} mb-3`}>
            {errorDetails.message}
          </p>
          
          {errorDetails.suggestions && errorDetails.suggestions.length > 0 && (
            <ul className={`${errorDetails.textColor} text-sm space-y-1 mb-4`}>
              {errorDetails.suggestions.map((suggestion, index) => (
                <li key={index} className="flex items-start space-x-2">
                  <span className="text-xs mt-1.5">•</span>
                  <span>{suggestion}</span>
                </li>
              ))}
            </ul>
          )}
          
          <div className="flex items-center space-x-3">
            {onRetry && (
              <button
                onClick={onRetry}
                className={`inline-flex items-center space-x-2 px-4 py-2 bg-white border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors`}
              >
                <RefreshCw className="w-4 h-4" />
                <span>Try Again</span>
              </button>
            )}
            
            {dismissible && onDismiss && (
              <button
                onClick={onDismiss}
                className="text-sm font-medium text-gray-600 hover:text-gray-800 transition-colors"
              >
                Dismiss
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ErrorMessage;
