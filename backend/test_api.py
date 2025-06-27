#!/usr/bin/env python3

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{BASE_URL}/model-info")
    print("Model Info:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_supported_tickers():
    """Test supported tickers endpoint"""
    response = requests.get(f"{BASE_URL}/supported-tickers")
    print("Supported Tickers:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_prediction(ticker="AAPL"):
    """Test prediction endpoint"""
    data = {"ticker": ticker}
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Prediction for {ticker}:")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code} - {response.text}")
    print()

if __name__ == "__main__":
    print("Testing Stock Prediction API...\n")
    
    try:
        test_health()
        test_model_info()
        test_supported_tickers()
        test_prediction("AAPL")
        test_prediction("GOOGL")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"Error: {e}")
