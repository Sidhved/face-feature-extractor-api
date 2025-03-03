# ---------------------------------------------FastAPI Main Application-------------------------------------------- #

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from app.core.config import settings
from app.core.logging import logger
from app.api.routes import landmarks
from app.services.landmark_detection import landmark_detector

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
class RateLimitMiddleware:
    def __init__(self, app):
        self.app = app
        self.requests = {}
        self.rate_limit = settings.RATE_LIMIT
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        # Get client IP
        client_ip = scope["client"][0]
        current_time = time.time()
        
        # Clean up old requests
        self.requests = {ip: times for ip, times in self.requests.items() 
                        if times[-1] > current_time - 60}
        
        # Check rate limit
        if client_ip in self.requests:
            request_times = self.requests[client_ip]
            request_times = [t for t in request_times if t > current_time - 60]
            
            if len(request_times) >= self.rate_limit:
                # Rate limit exceeded
                response = JSONResponse(
                    status_code=429,
                    content={
                        "status": "error",
                        "message": f"Rate limit exceeded. Maximum {self.rate_limit} requests per minute."
                    }
                )
                await response(scope, receive, send)
                return
                
            # Update request times
            self.requests[client_ip] = request_times + [current_time]
        else:
            # First request from this IP
            self.requests[client_ip] = [current_time]
            
        await self.app(scope, receive, send)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Exception handler for HTTP exceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail
        }
    )

# Exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    import traceback
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error"
        }
    )

# Include routers
app.include_router(
    landmarks.router,
    prefix=f"{settings.API_V1_STR}/landmarks",
    tags=["landmarks"]
)

@app.get("/", tags=["status"])
async def root():
    """Root endpoint with API information."""
    return {
        "status": "ok",
        "api_name": settings.PROJECT_NAME,
        "api_version": "v1",
        "documentation_url": "/docs"
    }

@app.get("/health", tags=["status"])
async def health_check():
    """Health check endpoint."""
    # Check if landmark detector is initialized
    if not landmark_detector.initialized:
        landmark_detector.initialize()
    
    return {
        "status": "healthy",
        "dlib_initialized": landmark_detector.initialized
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

# ---------------------------------------------------------------------------------------------------- #