#!/usr/bin/env python3
"""
Advanced LSTM Stock Price Prediction Training Script
Features:
- Multiple technical indicators
- Advanced LSTM architecture with attention mechanism
- Comprehensive feature engineering
- Multi-company training data
- Production-ready model export
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
import warnings
import pickle
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """Class for computing technical indicators"""
    
    @staticmethod
    def sma(data, window):
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data, window=14):
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        # Avoid division by zero
        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        # Clip RSI to valid range
        return rsi.clip(0, 100)
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD indicator"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line, macd_line - signal_line
    
    @staticmethod
    def bollinger_bands(data, window=20, std_dev=2):
        """Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def stochastic_oscillator(high, low, close, k_window=14, d_window=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        # Avoid division by zero
        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, 1e-10)
        k_percent = 100 * ((close - lowest_low) / denominator)
        k_percent = k_percent.clip(0, 100)  # Ensure valid range
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent


class StockDataProcessor:
    """Process stock data and add technical indicators"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
    
    def fetch_data(self, tickers, start_date, end_date):
        """Fetch stock data for multiple tickers"""
        data = {}
        for ticker in tickers:
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not stock_data.empty:
                    stock_data.dropna(inplace=True)
                    data[ticker] = stock_data
                    print(f"Downloaded {len(stock_data)} records for {ticker}")
                else:
                    print(f"No data found for {ticker}")
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
        return data
    
    def engineer_features(self, df):
        """Add comprehensive technical indicators"""
        df = df.copy()
        
        # Ensure we're working with proper column access for multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten column names if they are MultiIndex
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Basic price features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Moving averages
        df['SMA_10'] = TechnicalIndicators.sma(df['Close'], 10)
        df['SMA_20'] = TechnicalIndicators.sma(df['Close'], 20)
        df['SMA_50'] = TechnicalIndicators.sma(df['Close'], 50)
        df['EMA_12'] = TechnicalIndicators.ema(df['Close'], 12)
        df['EMA_26'] = TechnicalIndicators.ema(df['Close'], 26)
        
        # RSI
        df['RSI'] = TechnicalIndicators.rsi(df['Close'])
        
        # MACD
        macd, signal, histogram = TechnicalIndicators.macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = TechnicalIndicators.stochastic_oscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_d
        
        # Volume indicators - ensure proper Series operations
        volume_sma = TechnicalIndicators.sma(df['Volume'], 20)
        df['Volume_SMA'] = volume_sma
        df['Volume_Ratio'] = df['Volume'] / volume_sma
        
        # Price position indicators
        df['Close_to_High'] = df['Close'] / df['High']
        df['Close_to_Low'] = df['Close'] / df['Low']
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Forward fill and backward fill NaN values
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Replace infinite values with NaN and then fill them
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Final check - fill any remaining NaN with 0
        df.fillna(0, inplace=True)
        
        return df
    
    def prepare_sequences(self, data, sequence_length, target_column='Close'):
        """Prepare sequences for LSTM training"""
        sequences = []
        targets = []
        
        # Convert DataFrame to numpy array for easier indexing
        if isinstance(data, pd.DataFrame):
            feature_cols = [col for col in data.columns if col != target_column]
            target_col_idx = data.columns.get_loc(target_column)
            
            # Separate features and target
            feature_data = data[feature_cols].values
            target_data = data[target_column].values
            
            for i in range(len(data) - sequence_length):
                # Get sequence of features
                seq = feature_data[i:i + sequence_length]
                # Get target value
                target = target_data[i + sequence_length]
                sequences.append(seq)
                targets.append(target)
        else:
            # Handle numpy array case
            for i in range(len(data) - sequence_length):
                seq = data[i:i + sequence_length, :-1]  # Exclude last column (target)
                target = data[i + sequence_length, -1]  # Last column is target
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)


class StockDataset(Dataset):
    """Custom Dataset for stock data"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class StockLSTM(nn.Module):
    """Advanced LSTM model with attention mechanism"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layers with regularization
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size // 2)
        
    def attention_mechanism(self, lstm_output):
        """Apply attention mechanism"""
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        return attended_output
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        attended_out = self.attention_mechanism(lstm_out)
        
        # Forward pass through dense layers
        out = self.dropout(attended_out)
        out = F.relu(self.fc1(out))
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def main():
    """Main training function"""
    # Configuration
    TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'CRM']
    START_DATE = '2015-01-01'
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    SEQUENCE_LENGTH = 60
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 10
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    DROPOUT = 0.2
    
    print("Starting LSTM Stock Prediction Training...")
    print(f"Training on {len(TICKERS)} companies from {START_DATE} to {END_DATE}")
    
    # Initialize processor
    processor = StockDataProcessor()
    
    # Fetch and process data
    print("\nFetching stock data...")
    raw_data = processor.fetch_data(TICKERS, START_DATE, END_DATE)
    
    if not raw_data:
        print("No data fetched. Exiting.")
        return
    
    # Combine all data
    print("\nEngineering features...")
    all_sequences = []
    all_targets = []
    
    feature_columns = None
    scaler = MinMaxScaler()
    
    for ticker, df in raw_data.items():
        # Add technical indicators
        df_processed = processor.engineer_features(df)
        
        # Select feature columns (excluding target)
        if feature_columns is None:
            feature_columns = [col for col in df_processed.columns if col not in ['Close']]
            print(f"Using {len(feature_columns)} features: {feature_columns[:5]}...")
        
        # Scale features
        features_scaled = scaler.fit_transform(df_processed[feature_columns + ['Close']])
        df_scaled = pd.DataFrame(features_scaled, columns=feature_columns + ['Close'])
        
        # Create sequences
        sequences, targets = processor.prepare_sequences(
            df_scaled, SEQUENCE_LENGTH, target_column='Close'
        )
        
        all_sequences.extend(sequences)
        all_targets.extend(targets)
        
        print(f"Added {len(sequences)} sequences from {ticker}")
    
    # Convert to numpy arrays
    all_sequences = np.array(all_sequences)
    all_targets = np.array(all_targets)
    
    print(f"\nTotal training sequences: {len(all_sequences)}")
    print(f"Feature dimensions: {all_sequences.shape}")
    
    # Create dataset and dataloader
    dataset = StockDataset(all_sequences, all_targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    input_size = len(feature_columns)
    model = StockLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    print(f"\nModel architecture:")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {HIDDEN_SIZE}")
    print(f"Number of layers: {NUM_LAYERS}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = 0
        
        for batch_sequences, batch_targets in dataloader:
            batch_sequences = batch_sequences.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            outputs = model(batch_sequences)
            loss = criterion(outputs.squeeze(), batch_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        
        # Print completion message for each epoch
        print(f"Epoch {epoch + 1} done, average loss: {avg_loss:.6f}")
    
    # Save model and components
    print("\nSaving model and components...")
    
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': input_size,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT
        },
        'feature_columns': feature_columns,
        'sequence_length': SEQUENCE_LENGTH
    }, 'lstm_stock_predictor.pth')
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'tickers': TICKERS,
        'total_sequences': len(all_sequences),
        'epochs': EPOCHS,
        'final_loss': avg_loss,
        'feature_count': len(feature_columns)
    }
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\n=== Training Complete ===")
    print(f"Model saved to: lstm_stock_predictor.pth")
    print(f"Scaler saved to: scaler.pkl")
    print(f"Metadata saved to: model_metadata.pkl")
    print(f"Final training loss: {avg_loss:.6f}")
    print(f"Total training sequences: {len(all_sequences)}")
    print(f"Features used: {len(feature_columns)}")


if __name__ == "__main__":
    main()

