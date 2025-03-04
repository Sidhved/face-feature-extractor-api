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
RUN curl -L -c cookies.txt "https://drive.google.com/uc?export=download&id=1A9g08oIQAEaxOshvMc6Hb6DPX6RQ-JiX" > confirmation.html && \
    CONFIRM=$(cat confirmation.html | grep -o 'confirm=[^&]*' | cut -d'=' -f2) && \
    curl -L -b cookies.txt "https://drive.google.com/uc?export=download&id=1A9g08oIQAEaxOshvMc6Hb6DPX6RQ-JiX" -o shape_predictor_68_face_landmarks.dat && \
    rm cookies.txt confirmation.html

# Create non-root user with explicit UID/GID in the required range
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/bash -m appuser

# Set ownership and permissions
RUN chown appuser:appuser /app/shape_predictor_68_face_landmarks.dat
RUN chown -R appuser:appuser /app /home/appuser
RUN chmod -R 755 /app

# Explicitly set the USER directive with the UID
USER 10001

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
