from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import requests
import os
from dotenv import load_dotenv
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="AI Image Generation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STARRYAI_API_URL = "https://api.starryai.com/creations/"
STARRYAI_API_KEY = os.environ.get('STARRYAI_API_KEY')

class ImageRequest(BaseModel):
    prompt: str 
    negative_prompt: Optional[str] = None
    aspect_ratio: Optional[str] = "square"
    high_resolution: Optional[bool] = False
    num_images: Optional[int] = 4
    steps: Optional[int] = 20
    initial_image_mode: Optional[str] = "color"

class ImageResponse(BaseModel):
    status: str
    message: str
    generation_id: Optional[str] = None
    urls: Optional[list] = None

@app.post("/generate", response_model=ImageResponse)
async def generate_image(request: ImageRequest):
    try:
        if not STARRYAI_API_KEY:
            raise HTTPException(status_code=500, detail="API key not configured")


        payload = {
            "prompt": request.prompt, 
            "model": "lyra",
            "aspectRatio": request.aspect_ratio,
            "highResolution": request.high_resolution,
            "images": request.num_images,
            "steps": request.steps,
            "initialImageMode": request.initial_image_mode,
            # "negativePrompt": request.negative_prompt if hasattr(request, 'negative_prompt') else None
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-API-Key": STARRYAI_API_KEY
        }

        logger.info(f"Sending request to StarryAI with payload: {payload}")
        
        response = requests.post(STARRYAI_API_URL, json=payload, headers=headers)
        
        logger.info(f"StarryAI Response Status: {response.status_code}")
        logger.info(f"StarryAI Response: {response.text}")

        if response.status_code == 200:
            data = response.json()
            return ImageResponse(
                status="success",
                message="Image generation started",
                generation_id=data.get("id"),
                urls=[]
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"StarryAI API error: {response.text}"
            )

    except Exception as e:
        logger.error(f"Error in generate_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{generation_id}")
async def get_generation_status(generation_id: str):
    try:
        headers = {
            "accept": "application/json",
            "X-API-Key": STARRYAI_API_KEY
        }
        
        response = requests.get(
            f"{STARRYAI_API_URL}{generation_id}",
            headers=headers
        )
        
        logger.info(f"Status Check Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            
            images = data.get("images", [])
            image_urls = [img.get("url") for img in images if img.get("url")]
            
            status = "pending"
            if image_urls:
                status = "completed"
            elif data.get("status") == "failed":
                status = "failed"
                
            return {
                "status": status,
                "urls": image_urls,
                "raw_status": data.get("status", "unknown")
            }
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error checking status: {response.text}"
            )

    except Exception as e:
        logger.error(f"Error in get_generation_status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 