import React, { useEffect, useRef } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';
import type { CandlestickData } from '../types';

interface StockChartProps {
  data: CandlestickData[];
  height?: number;
}

const StockChart: React.FC<StockChartProps> = ({ data, height = 400 }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height,
      layout: {
        background: { color: '#ffffff' },
        textColor: '#333',
      },
      grid: {
        vertLines: {
          color: '#eee',
        },
        horzLines: {
          color: '#eee',
        },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: '#ccc',
      },
      timeScale: {
        borderColor: '#ccc',
      },
    });

    const candleSeries = chart.addCandlestickSeries();

    candleSeries.setData(
      data.map(item => ({
        time: item.time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }))
    );

    return () => {
      chart.remove();
    };
  }, [data, height]);

  return <div ref={chartContainerRef} style={{ position: 'relative', width: '100%' }} />;
};

export default StockChart;

