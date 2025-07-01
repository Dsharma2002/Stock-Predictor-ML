FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install Python dependencies
RUN uv sync --frozen

# Create models directory
RUN mkdir -p ./models

# Copy application code
COPY . .

# Update model paths in the application to use mounted volume
RUN sed -i 's|../training/|./models/|g' stock_predictor_api.py

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "stock_predictor_api:app", "--host", "0.0.0.0", "--port", "8000"]
