import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Search, TrendingUp, X } from 'lucide-react';
import { validateTicker, formatTicker } from '../utils/formatters';

// Comprehensive stock tickers database
const STOCK_DATABASE = [
  // Tech Giants
  { symbol: 'AAPL', name: 'Apple Inc.' },
  { symbol: 'GOOGL', name: 'Alphabet Inc. (Class A)' },
  { symbol: 'GOOG', name: 'Alphabet Inc. (Class C)' },
  { symbol: 'MSFT', name: 'Microsoft Corporation' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.' },
  { symbol: 'META', name: 'Meta Platforms Inc.' },
  { symbol: 'TSLA', name: 'Tesla Inc.' },
  { symbol: 'NVDA', name: 'NVIDIA Corporation' },
  { symbol: 'NFLX', name: 'Netflix Inc.' },
  { symbol: 'AMD', name: 'Advanced Micro Devices Inc.' },
  
  // Cloud & Software
  { symbol: 'CRM', name: 'Salesforce Inc.' },
  { symbol: 'ORCL', name: 'Oracle Corporation' },
  { symbol: 'ADBE', name: 'Adobe Inc.' },
  { symbol: 'NOW', name: 'ServiceNow Inc.' },
  { symbol: 'INTU', name: 'Intuit Inc.' },
  { symbol: 'WDAY', name: 'Workday Inc.' },
  { symbol: 'TEAM', name: 'Atlassian Corporation' },
  { symbol: 'ZM', name: 'Zoom Video Communications' },
  { symbol: 'OKTA', name: 'Okta Inc.' },
  { symbol: 'SNOW', name: 'Snowflake Inc.' },
  
  // Financial Services
  { symbol: 'JPM', name: 'JPMorgan Chase & Co.' },
  { symbol: 'BAC', name: 'Bank of America Corporation' },
  { symbol: 'WFC', name: 'Wells Fargo & Company' },
  { symbol: 'GS', name: 'Goldman Sachs Group Inc.' },
  { symbol: 'MS', name: 'Morgan Stanley' },
  { symbol: 'V', name: 'Visa Inc.' },
  { symbol: 'MA', name: 'Mastercard Incorporated' },
  { symbol: 'PYPL', name: 'PayPal Holdings Inc.' },
  { symbol: 'SQ', name: 'Block Inc.' },
  { symbol: 'COIN', name: 'Coinbase Global Inc.' },
  
  // E-commerce & Consumer
  { symbol: 'SHOP', name: 'Shopify Inc.' },
  { symbol: 'EBAY', name: 'eBay Inc.' },
  { symbol: 'ETSY', name: 'Etsy Inc.' },
  { symbol: 'WMT', name: 'Walmart Inc.' },
  { symbol: 'TGT', name: 'Target Corporation' },
  { symbol: 'COST', name: 'Costco Wholesale Corporation' },
  { symbol: 'HD', name: 'Home Depot Inc.' },
  { symbol: 'LOW', name: 'Lowe\'s Companies Inc.' },
  
  // Transportation & Mobility
  { symbol: 'UBER', name: 'Uber Technologies Inc.' },
  { symbol: 'LYFT', name: 'Lyft Inc.' },
  { symbol: 'DASH', name: 'DoorDash Inc.' },
  { symbol: 'F', name: 'Ford Motor Company' },
  { symbol: 'GM', name: 'General Motors Company' },
  { symbol: 'RIVN', name: 'Rivian Automotive Inc.' },
  { symbol: 'LCID', name: 'Lucid Group Inc.' },
  
  // Media & Entertainment
  { symbol: 'DIS', name: 'Walt Disney Company' },
  { symbol: 'CMCSA', name: 'Comcast Corporation' },
  { symbol: 'T', name: 'AT&T Inc.' },
  { symbol: 'VZ', name: 'Verizon Communications Inc.' },
  { symbol: 'SPOT', name: 'Spotify Technology S.A.' },
  { symbol: 'ROKU', name: 'Roku Inc.' },
  
  // Healthcare & Biotech
  { symbol: 'JNJ', name: 'Johnson & Johnson' },
  { symbol: 'PFE', name: 'Pfizer Inc.' },
  { symbol: 'MRNA', name: 'Moderna Inc.' },
  { symbol: 'BNTX', name: 'BioNTech SE' },
  { symbol: 'GILD', name: 'Gilead Sciences Inc.' },
  { symbol: 'AMGN', name: 'Amgen Inc.' },
  { symbol: 'BIIB', name: 'Biogen Inc.' },
  
  // Semiconductors
  { symbol: 'INTC', name: 'Intel Corporation' },
  { symbol: 'QCOM', name: 'QUALCOMM Incorporated' },
  { symbol: 'AVGO', name: 'Broadcom Inc.' },
  { symbol: 'TXN', name: 'Texas Instruments Incorporated' },
  { symbol: 'AMAT', name: 'Applied Materials Inc.' },
  { symbol: 'LRCX', name: 'Lam Research Corporation' },
  { symbol: 'KLAC', name: 'KLA Corporation' },
  
  // Energy & Commodities
  { symbol: 'XOM', name: 'Exxon Mobil Corporation' },
  { symbol: 'CVX', name: 'Chevron Corporation' },
  { symbol: 'COP', name: 'ConocoPhillips' },
  { symbol: 'SLB', name: 'Schlumberger Limited' },
  { symbol: 'OXY', name: 'Occidental Petroleum Corporation' },
  
  // Real Estate & REITs
  { symbol: 'AMT', name: 'American Tower Corporation' },
  { symbol: 'PLD', name: 'Prologis Inc.' },
  { symbol: 'CCI', name: 'Crown Castle International Corp.' },
  { symbol: 'EQIX', name: 'Equinix Inc.' },
  { symbol: 'DLR', name: 'Digital Realty Trust Inc.' },
];

interface TickerInputProps {
  onPredict: (ticker: string) => void;
  isLoading: boolean;
  supportedTickers?: string[];
}

const TickerInput: React.FC<TickerInputProps> = ({ onPredict, isLoading, supportedTickers = [] }) => {
  const [ticker, setTicker] = useState('');
  const [error, setError] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsRef = useRef<HTMLDivElement>(null);

  // Filter suggestions based on input
  const filteredSuggestions = useMemo(() => {
    if (!ticker || ticker.length < 1) return [];
    
    const searchTerm = ticker.toLowerCase();
    return STOCK_DATABASE.filter(
      (stock) =>
        stock.symbol.toLowerCase().includes(searchTerm) ||
        stock.name.toLowerCase().includes(searchTerm)
    ).slice(0, 6); // Limit to 6 suggestions
  }, [ticker]);

  // Handle click outside to close suggestions
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        inputRef.current &&
        !inputRef.current.contains(event.target as Node) &&
        suggestionsRef.current &&
        !suggestionsRef.current.contains(event.target as Node)
      ) {
        setShowSuggestions(false);
        setSelectedIndex(-1);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const formattedTicker = formatTicker(ticker);
    
    if (!formattedTicker) {
      setError('Please enter a stock ticker');
      return;
    }
    
    if (!validateTicker(formattedTicker)) {
      setError('Please enter a valid stock ticker (1-5 letters)');
      return;
    }
    
    setError('');
    setShowSuggestions(false);
    onPredict(formattedTicker);
  };

  const handleTickerClick = (tickerSymbol: string) => {
    setTicker(tickerSymbol);
    setError('');
    setShowSuggestions(false);
    onPredict(tickerSymbol);
  };

  const handleInputChange = (value: string) => {
    setTicker(value.toUpperCase());
    setError('');
    setShowSuggestions(value.length > 0);
    setSelectedIndex(-1);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showSuggestions || filteredSuggestions.length === 0) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex((prev) => 
          prev < filteredSuggestions.length - 1 ? prev + 1 : prev
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
        break;
      case 'Enter':
        if (selectedIndex >= 0) {
          e.preventDefault();
          handleTickerClick(filteredSuggestions[selectedIndex].symbol);
        }
        break;
      case 'Escape':
        setShowSuggestions(false);
        setSelectedIndex(-1);
        break;
    }
  };

  return (
    <div className="w-full max-w-md mx-auto">
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="flex space-x-3">
          <div className="relative flex-1">
            <input
              ref={inputRef}
              type="text"
              value={ticker}
              onChange={(e) => handleInputChange(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => setShowSuggestions(ticker.length > 0)}
              placeholder="Enter stock ticker (e.g., AAPL)"
              className={`w-full px-4 py-3 pl-12 text-lg border-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors ${
                error ? 'border-red-500' : 'border-gray-300 focus:border-blue-500'
              }`}
              disabled={isLoading}
              maxLength={5}
            />
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-6 h-6" />
            
            {ticker && (
              <button
                type="button"
                onClick={() => {
                  setTicker('');
                  setShowSuggestions(false);
                  setError('');
                  inputRef.current?.focus();
                }}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            )}
            
            {/* Suggestions Dropdown */}
            {showSuggestions && filteredSuggestions.length > 0 && (
              <div
                ref={suggestionsRef}
                className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-50 max-h-60 overflow-y-auto"
              >
                {filteredSuggestions.map((stock, index) => (
                  <button
                    key={stock.symbol}
                    type="button"
                    onClick={() => handleTickerClick(stock.symbol)}
                    className={`w-full px-4 py-3 text-left hover:bg-gray-50 transition-colors border-b border-gray-100 last:border-b-0 ${
                      index === selectedIndex ? 'bg-blue-50 text-blue-700' : 'text-gray-700'
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <div>
                        <div className="font-semibold">{stock.symbol}</div>
                        <div className="text-sm text-gray-500 truncate">{stock.name}</div>
                      </div>
                      <TrendingUp className="w-4 h-4 text-gray-400" />
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
          
          <button
            type="submit"
            disabled={isLoading || !ticker}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2 text-lg font-medium"
          >
            <TrendingUp className="w-5 h-5" />
            <span>Predict</span>
          </button>
        </div>
        {error && <p className="text-red-600 text-sm mt-2">{error}</p>}
      </form>

      {supportedTickers.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 mb-3">Supported Tickers:</h3>
          <div className="flex flex-wrap gap-2">
            {supportedTickers.map((tickerSymbol) => (
              <button
                key={tickerSymbol}
                onClick={() => handleTickerClick(tickerSymbol)}
                disabled={isLoading}
                className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {tickerSymbol}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TickerInput;
