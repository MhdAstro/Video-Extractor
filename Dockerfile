FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    HOST=0.0.0.0 \
    LOG_LEVEL=debug

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for temporary files
RUN mkdir -p /tmp/video_processing && \
    chmod 777 /tmp/video_processing

# Set environment variables for the application
ENV TMPDIR=/tmp/video_processing

# Expose port
EXPOSE 8000

# Create startup script
RUN echo '#!/bin/bash\n\
uvicorn main:app \
--host ${HOST} \
--port ${PORT} \
--workers 1 \
--timeout-keep-alive 300 \
--log-level ${LOG_LEVEL} \
--proxy-headers \
--forwarded-allow-ips "*"' > /app/start.sh && \
chmod +x /app/start.sh

# Command to run the application
CMD ["/app/start.sh"] 