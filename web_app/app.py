# app.py - Web interface for plant identifier

import os
import uvicorn
import uuid
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time
from typing import List
import shutil
from pathlib import Path
import sys

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import the plant identifier
from plant_identifier import PlantIdentifier

# Create the FastAPI app
app = FastAPI(title="Plant Identification System")

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Mount the static directory
app.mount("/static", StaticFiles(directory=os.path.join(script_dir, "static")), name="static")

# Set up templates
templates = Jinja2Templates(directory=os.path.join(script_dir, "templates"))

# Initialize the plant identifier once at startup (improves performance)
print("Initializing plant identifier model...")
plant_identifier = PlantIdentifier()
print("Model initialized successfully!")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):
    """Make prediction on uploaded image"""
    # Generate a unique filename to avoid collisions
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join("uploads", unique_filename)
    
    # Save the uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Error saving file: {str(e)}"}
        )
    
    # Make prediction
    start_time = time.time()
    result = plant_identifier.predict(file_path)
    processing_time = time.time() - start_time
    
    # Add additional info to result
    result["filename"] = file.filename
    result["processing_time"] = f"{processing_time:.2f} seconds"
    
    # Make sure the static directory exists before copying
    static_path = os.path.join("static", unique_filename)
    
    # Copy file to static directory for display
    try:
        shutil.copy(file_path, static_path)
        result["image_path"] = f"/static/{unique_filename}"
    except Exception as e:
        print(f"Error copying file to static directory: {str(e)}")
        # Still allow processing if copying fails
        result["image_path"] = ""
    
    # Ensure required fields exist in the result dict
    if "error" in result and "confidence" not in result:
        result["is_plant"] = False
        result["confidence"] = 0.0
        result["top_predictions"] = []
    
    # Return template with results
    return templates.TemplateResponse("result.html", {
        "request": request,
        "result": result
    })

@app.post("/batch-predict/")
async def batch_predict(request: Request, files: List[UploadFile] = File(...)):
    """Make predictions on multiple uploaded images"""
    results = []
    
    for file in files:
        # Generate a unique filename to avoid collisions
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join("uploads", unique_filename)
        
        try:
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Make prediction
            start_time = time.time()
            result = plant_identifier.predict(file_path)
            processing_time = time.time() - start_time
            
            # Copy file to static directory for display
            static_path = os.path.join("static", unique_filename)
            shutil.copy(file_path, static_path)
            
            # Add additional info to result
            result["filename"] = file.filename
            result["processing_time"] = f"{processing_time:.2f} seconds"
            result["image_path"] = f"/static/{unique_filename}"
            
            # Ensure required fields exist in the result dict
            if "error" in result and "confidence" not in result:
                result["is_plant"] = False
                result["confidence"] = 0.0
                result["top_predictions"] = []
            
            results.append(result)
            
        except Exception as e:
            # Handle any errors in individual file processing
            print(f"Error processing file {file.filename}: {str(e)}")
            results.append({
                "error": str(e),
                "is_plant": False,
                "confidence": 0.0,
                "filename": file.filename,
                "processing_time": "0.00 seconds",
                "image_path": "",
                "top_predictions": []
            })
    
    # Return template with results
    return templates.TemplateResponse("batch_result.html", {
        "request": request,
        "results": results
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "model_loaded": plant_identifier is not None}

# Main entry point
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)