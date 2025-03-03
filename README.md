# Facial Feature Extractor API

A robust FastAPI-based service for extracting and analyzing facial landmarks from images using dlib.

## Overview

This API provides endpoints for detecting facial landmarks in images, which can be used for various applications such as:
- Facial recognition
- Emotion detection
- Face morphing
- Virtual try-on systems
- Medical analysis

The service uses dlib's 68-point facial landmark detector to identify key facial features with high accuracy.

## Features

- **Facial Landmark Detection**: Extract 68 facial landmarks from images
- **Rate Limiting**: Built-in protection against API abuse
- **Error Handling**: Comprehensive error responses
- **Logging**: Detailed logging for monitoring and debugging
- **Health Checks**: Endpoint for monitoring service health
- **CORS Support**: Cross-Origin Resource Sharing enabled for web applications

## Tech Stack

- **FastAPI**: Modern, high-performance web framework
- **dlib**: Machine learning toolkit with facial landmark detection
- **OpenCV**: Computer vision library for image processing
- **Pydantic**: Data validation and settings management
- **Uvicorn**: ASGI server for serving the API

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Sidhved/face-feature-extractor-api.git
   cd facial_feature_api
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .env
   source .env/bin/activate  # On Windows: .env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the shape predictor model:
   ```bash
   # The shape_predictor_68_face_landmarks.dat file should be placed in the facial_feature_api_dlib directory
   # You can download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2(http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   ```

## Usage

### Starting the API

```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

### API Documentation

Once the server is running, you can access:
- Interactive API documentation: http://localhost:8000/docs
- Alternative documentation: http://localhost:8000/redoc

### Example API Calls

#### Detect Facial Landmarks

```bash
curl -X POST "http://localhost:8000/api/v1/landmarks/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/your/image.jpg"
```

## Project Structure

facial-feature-extractor-api/
├── app/
│ ├── api/ # API routes and endpoints
│ ├── core/ # Core configurations and settings
│ ├── services/ # Business logic and services
│ ├── utils/ # Utility functions
│ └── main.py # FastAPI application entry point
├── tests/ # Test suite
│ ├── integration/ # Integration tests
│ └── unit/ # Unit tests
├── logs/ # Application logs
├── results/ # Output results directory
├── .env/ # Virtual environment (not tracked in git)
├── requirements.txt # Project dependencies
└── shape_predictor_68_face_landmarks.dat # dlib model file


## Testing

Run the test suite with:

```bash
pytest tests
```

## Configuration

The application can be configured through environment variables or a `.env` file. See `app/core/config.py` for available settings.

## License

[MIT License](LICENSE)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

Sidhved Warik - sidhved.warik@gmail.com

Project Link: [https://github.com/Sidhved/face-feature-extractor-api](https://github.com/Sidhved/face-feature-extractor-api.git)
