from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles # Import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Assuming your routers are in these files
from auth import router as auth_router
from inference import router as inference_router

app = FastAPI()

# 1. Set up CORS middleware to allow requests from your frontend
origins = [
    "http://localhost:3000",  # The origin for your Next.js app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Mount the static directory to serve video files
# This makes the 'static' folder accessible to the browser at the path '/static'
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. Include your API routers
app.include_router(auth_router, prefix="/auth")
app.include_router(inference_router, prefix="/video")

# 4. Define a root endpoint for health checks
@app.get("/")
def root():
    return {"message": "Backend is running correctly"}

