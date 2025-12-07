# Use your chosen Python image
FROM python:3.12.6-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Install CPU-only PyTorch (Keep this to save space)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# --- NEW CHANGES START HERE ---
# Expose the default Streamlit port
EXPOSE 7860
# Healthcheck (Optional but good for robustness)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app using Streamlit, not Python
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
# --- NEW CHANGES END HERE ---
