FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
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
# Uncomment if you need to download the model during build
RUN curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" | bunzip2 > shape_predictor_68_face_landmarks.dat

# Create a non-root user
RUN groupadd -r appuser --gid 10001 && useradd --uid 10001 -r -g appuser appuser

# Set ownership and permissions
RUN chown -R appuser:appuser /app
RUN chmod -R 755 /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
