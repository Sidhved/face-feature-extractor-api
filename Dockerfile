FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for dlib and curl for downloading
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    curl \
    bzip2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directories for logs and results
RUN mkdir -p logs results

# Download the shape predictor model if not included in the repository
RUN curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" | bunzip2 > shape_predictor_68_face_landmarks.dat

# Create non-root user with explicit UID/GID in the required range
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/bash -m appuser

# Set ownership and permissions
RUN chown -R appuser:appuser /app /home/appuser
RUN chmod -R 755 /app

# Explicitly set the USER directive with the UID
USER 10001

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
