import React from 'react';
import { Newspaper, ExternalLink, Clock } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import type { NewsItem } from '../types';

interface StockNewsProps {
  news: NewsItem[];
  ticker: string;
  isLoading?: boolean;
}

const StockNews: React.FC<StockNewsProps> = ({ news, ticker, isLoading = false }) => {
  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
        <div className="flex items-center space-x-2 mb-4">
          <Newspaper className="w-6 h-6 text-blue-600" />
          <h3 className="text-xl font-semibold text-gray-800">Latest News</h3>
        </div>
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-1/2 mb-2"></div>
              <div className="h-3 bg-gray-200 rounded w-full"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (news.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
        <div className="flex items-center space-x-2 mb-4">
          <Newspaper className="w-6 h-6 text-blue-600" />
          <h3 className="text-xl font-semibold text-gray-800">Latest News</h3>
        </div>
        <div className="text-center py-8">
          <Newspaper className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500">No recent news available for {ticker}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
      <div className="flex items-center space-x-2 mb-6">
        <Newspaper className="w-6 h-6 text-blue-600" />
        <h3 className="text-xl font-semibold text-gray-800">Latest News for {ticker}</h3>
      </div>
      
      <div className="space-y-6">
        {news.map((item, index) => (
          <div key={index} className="border-b border-gray-100 last:border-b-0 pb-4 last:pb-0">
            <a
              href={item.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group block hover:bg-gray-50 -mx-2 px-2 py-2 rounded-lg transition-colors cursor-pointer"
              onClick={(e) => {
                e.preventDefault();
                window.open(item.url, '_blank', 'noopener,noreferrer');
              }}
            >
              <div className="flex items-start justify-between mb-2">
                <h4 className="text-lg font-medium text-gray-900 group-hover:text-blue-600 transition-colors leading-tight pr-4">
                  {item.title}
                </h4>
                <ExternalLink className="w-4 h-4 text-gray-400 group-hover:text-blue-600 flex-shrink-0 mt-1" />
              </div>
              
              <div className="flex items-center space-x-4 text-sm text-gray-500 mb-2">
                <span className="font-medium">{item.source}</span>
                <div className="flex items-center space-x-1">
                  <Clock className="w-3 h-3" />
                  <span>
                    {formatDistanceToNow(new Date(item.published_at), { addSuffix: true })}
                  </span>
                </div>
              </div>
              
              {item.summary && (
                <p className="text-gray-600 text-sm leading-relaxed">
                  {item.summary}
                </p>
              )}
            </a>
          </div>
        ))}
      </div>
      
      <div className="mt-6 pt-4 border-t border-gray-100">
        <p className="text-xs text-gray-500 text-center">
          News powered by Yahoo Finance
        </p>
      </div>
    </div>
  );
};

export default StockNews;
