FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Default command (runs the Streamlit dashboard)
CMD ["streamlit", "run", "src/dashboard_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Expose the port Streamlit runs on
EXPOSE 8501
