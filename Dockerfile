FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (OpenCV / GL support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, and assets
COPY . .

# Hugging Face Spaces requires running as User ID 1000
RUN useradd -m -u 1000 user && \
    mkdir -p models database assets && \
    chown -R user:user /app && \
    chmod +x /app/start.sh

USER user

# Expose port 7860 for HF Spaces
EXPOSE 7860

CMD ["/app/start.sh"]