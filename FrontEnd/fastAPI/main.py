from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
from supabase import create_client, Client
import uvicorn
import cloudinary
import cloudinary.uploader
from PIL import Image
import io
import uuid
from datetime import datetime
from typing import Optional
import aiohttp
import traceback
import sys

# Manually load .env file with UTF-8 encoding
try:
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    with open(env_path, 'r', encoding='utf-8') as env_file:
        env_content = env_file.read()
        print(".env file content:")
        print(env_content)  # Debugging: Print the content of the .env file
        for line in env_content.splitlines():
            if line.strip() and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    print("Environment variables loaded successfully.")
except UnicodeDecodeError as e:
    print(f"Error decoding .env file: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error loading .env file: {e}")
    sys.exit(1)

# Check for required environment variables
required_env_vars = [
    "SUPABASE_URL", 
    "SUPABASE_ANON_KEY",
    "CLOUDINARY_CLOUD_NAME",
    "CLOUDINARY_API_KEY",
    "CLOUDINARY_API_SECRET"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
    print("Please create a .env file with these variables")
    # For development, set some defaults if environment variables are missing
    if "SUPABASE_URL" in missing_vars:
        os.environ["SUPABASE_URL"] = "https://your-project.supabase.co"
        print("Using placeholder SUPABASE_URL. Please set the correct value.")
    if "SUPABASE_ANON_KEY" in missing_vars:
        os.environ["SUPABASE_ANON_KEY"] = "your-anon-key"
        print("Using placeholder SUPABASE_ANON_KEY. Please set the correct value.")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
try:
    supabase_url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    
    print(f"Connecting to Supabase at {supabase_url}")
    supabase: Client = create_client(supabase_url, supabase_key)
    print("Supabase client created successfully")
except Exception as e:
    print(f"Error initializing Supabase client: {str(e)}")
    supabase = None

# Configure Cloudinary
try:
    print("Configuring Cloudinary...")
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET")
    )
    print("Cloudinary configured successfully")
except Exception as e:
    print(f"Error configuring Cloudinary: {str(e)}")

async def validate_image(file: UploadFile) -> bool:
    if not file.content_type.startswith('image/'):
        return False
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        await file.seek(0)  # Reset file pointer
        return True
    except Exception:
        return False

async def process_and_upload_image(image: UploadFile, prefix: str) -> str:
    content = await image.read()
    filename = f"{prefix}_{uuid.uuid4()}"
    
    result = cloudinary.uploader.upload(content, 
        public_id=filename,
        folder="virtual-try-on"
    )
    return result['secure_url']

async def run_ai_model(user_image_url: str, cloth_image_url: str, cloth_type: str) -> Optional[str]:
    try:
        # Check for GPU endpoint URL
        gpu_endpoint = os.getenv("GPU_ENDPOINT_URL")
        
        if gpu_endpoint:
            # Use remote GPU endpoint if available
            print(f"Using remote GPU endpoint at: {gpu_endpoint}")
            try:
                # Prepare payload with image URLs and cloth type
                payload = {
                    "user_image_url": user_image_url,
                    "cloth_image_url": cloth_image_url,
                    "cloth_type": cloth_type
                }
                  # Send request to GPU endpoint
                async with aiohttp.ClientSession() as session:
                    print(f"Sending to GPU endpoint with cloth_type: {cloth_type}")
                    async with session.post(gpu_endpoint, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"GPU endpoint error: {response.status}, {error_text}")
                            raise Exception(f"GPU processing failed with status {response.status}")
                        
                        result_data = await response.json()
                        
                        if "result_image_url" in result_data:
                            print("Successfully received result from GPU endpoint")
                            return result_data["result_image_url"]
                        else:
                            print(f"Unexpected response format: {result_data}")
                            # Fall through to local processing if response format is incorrect
            
            except Exception as e:
                print(f"Error connecting to GPU endpoint: {str(e)}")
                print("Falling back to local image processing")
                # Fall through to local processing
        
        # Local processing (fallback or default if no GPU endpoint configured)
        print("Using local image processing (no AI model)")
        
        # Download images from URLs
        async with aiohttp.ClientSession() as session:
            async with session.get(user_image_url) as response:
                user_image_content = await response.read()
            async with session.get(cloth_image_url) as response:
                cloth_image_content = await response.read()

        try:
            # Process images using simple compositing (no AI model)
            user_img = Image.open(io.BytesIO(user_image_content)).convert('RGBA')
            cloth_img = Image.open(io.BytesIO(cloth_image_content)).convert('RGBA')
            
            # Make sure both images have alpha channel
            if user_img.mode != 'RGBA':
                user_img = user_img.convert('RGBA')
            if cloth_img.mode != 'RGBA':
                cloth_img = cloth_img.convert('RGBA')
                
            # Resize cloth image to match user image size
            cloth_img = cloth_img.resize(user_img.size)
            
            # Simple alpha blending instead of direct paste with mask
            # Create a new blank image with alpha channel
            result_img = Image.new('RGBA', user_img.size, (0, 0, 0, 0))
            
            # Paste the user image first
            result_img = Image.alpha_composite(result_img, user_img)
            
            # Then paste the clothing image on top with its alpha channel
            # Apply some transparency to the clothing for a more blended look
            cloth_data = cloth_img.getdata()
            new_data = []
            for item in cloth_data:
                # Make clothing slightly transparent
                # Keep color, but reduce alpha to 60%
                if item[3] > 0:  # If it's not fully transparent
                    new_data.append((item[0], item[1], item[2], int(item[3] * 0.6)))
                else:
                    new_data.append(item)  # Keep fully transparent pixels
            
            cloth_img.putdata(new_data)
            result_img = Image.alpha_composite(result_img, cloth_img)
            
            # Convert to RGB for saving as JPEG (removes alpha channel)
            result_img = result_img.convert('RGB')
            
            # Save to buffer
            buffer = io.BytesIO()
            result_img.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            
        except Exception as img_error:
            print(f"Error in image processing: {str(img_error)}")
            # Create a simple side-by-side image instead
            try:
                print("Trying alternative side-by-side compositing")
                user_img = Image.open(io.BytesIO(user_image_content)).convert('RGB')
                cloth_img = Image.open(io.BytesIO(cloth_image_content)).convert('RGB')
                
                # Resize both images to the same height
                height = min(user_img.height, cloth_img.height)
                user_width = int(user_img.width * (height / user_img.height))
                cloth_width = int(cloth_img.width * (height / cloth_img.height))
                
                user_img = user_img.resize((user_width, height))
                cloth_img = cloth_img.resize((cloth_width, height))
                
                # Create a side-by-side image
                result_img = Image.new('RGB', (user_width + cloth_width, height))
                result_img.paste(user_img, (0, 0))
                result_img.paste(cloth_img, (user_width, 0))
                
                # Save to buffer
                buffer = io.BytesIO()
                result_img.save(buffer, format='JPEG', quality=95)
                buffer.seek(0)
            except Exception as fallback_error:
                print(f"Fallback compositing also failed: {str(fallback_error)}")
                raise
        
        # Upload result to Cloudinary
        result = cloudinary.uploader.upload(
            buffer,
            folder="virtual-try-on/results",
            resource_type="image"
        )
        
        return result['secure_url']
    except Exception as e:
        print(f"Error in AI model processing: {str(e)}")
        return None

async def process_try_on(
    user_image_url: str,
    cloth_image_url: str,
    cloth_type: str,
    metadata_id: str,
    background_tasks: BackgroundTasks
):
    try:
        # Update status to processing
        supabase.table("try_on_history").update({
            "status": "processing"
        }).eq("id", metadata_id).execute()

        # Run AI model
        result_image_url = await run_ai_model(user_image_url, cloth_image_url, cloth_type)

        if not result_image_url:
            raise Exception("Failed to process images")

        # Update record with result
        supabase.table("try_on_history").update({
            "result_image_url": result_image_url,
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat()
        }).eq("id", metadata_id).execute()

    except Exception as e:
        # Update status to failed
        supabase.table("try_on_history").update({
            "status": "failed",
            "error": str(e)
        }).eq("id", metadata_id).execute()
        raise e

@app.post("/api/try-on")
async def try_on(
    background_tasks: BackgroundTasks,
    user_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    category_id: str = Form(...)
):
    try:
        # Extract category slug from product ID (e.g., "jeans-2" -> "jeans")
        category_slug = category_id
        
        # Define upper and lower category slugs
        upper_categories = ['t-shirts', 'full-sleeves', 'hoodies', 'polo']
        lower_categories = ['pants', 'shorts', 'cargo', 'jeans']
        
        # Determine cloth type based on category slug
        cloth_type = "upper" if category_slug in upper_categories else "lower"
        
        # Validate images
        print("Validating images...")
        if not await validate_image(user_image) or not await validate_image(cloth_image):
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Upload images to Cloudinary
        print("Uploading images to Cloudinary...")
        user_image_url = await process_and_upload_image(user_image, "user")
        cloth_image_url = await process_and_upload_image(cloth_image, "cloth")
    
        # Create initial record in database
        metadata = {
            "user_image_url": user_image_url,
            "cloth_image_url": cloth_image_url,
            "cloth_type": cloth_type,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }

        # Make metadata more visible in console
        print("\n" + "="*50)
        print("METADATA RECEIVED FROM UPLOAD:")
        print(f"USER IMAGE: {user_image_url}")
        print(f"CLOTH IMAGE: {cloth_image_url}")
        print(f"CLOTH TYPE: {cloth_type}")
        print(f"STATUS: pending")
        print(f"CREATED AT: {datetime.utcnow().isoformat()}")
        print("="*50 + "\n")
        
        # Insert metadata into database
        print("Inserting metadata into Supabase...")
        if not supabase:
            raise Exception("Supabase client not initialized properly")
            
        try:
            result = supabase.table("try_on_history").insert(metadata).execute()
            metadata_id = result.data[0]['id']
            print(f"INSERTED RECORD ID: {metadata_id}")
        except Exception as db_error:
            print(f"Database error: {str(db_error)}")
            print(f"Attempted to insert: {metadata}")
            # Return success anyway, just don't process in background
            return {
                "status": "success",
                "message": "Images processed, but database storage failed",
                "data": {
                    "status": "completed",
                    "user_image_url": user_image_url,
                    "cloth_image_url": cloth_image_url,
                    "cloth_type": cloth_type
                }
            }
          # Process try-on in background
        print("Starting background processing...")
        background_tasks.add_task(
            process_try_on,
            user_image_url,
            cloth_image_url,
            cloth_type,
            metadata_id,
            background_tasks
        )
        
        return {
            "status": "success",
            "message": "Try-on process started",
            "data": {
                "id": metadata_id,
                "status": "pending",
                "user_image_url": user_image_url,
                "cloth_image_url": cloth_image_url,
                "cloth_type": cloth_type
            }
        }
    except Exception as e:
        print(f"ERROR in try_on endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/try-on/{try_on_id}")
async def get_try_on_status(try_on_id: str):
    try:
        result = supabase.table("try_on_history").select("*").eq("id", try_on_id).execute()
        
        if not result.data:
            print(f"Record not found for ID: {try_on_id}")
            raise HTTPException(status_code=404, detail="Try-on record not found")
        
        print(f"Status check for {try_on_id}: {result.data[0].get('status', 'unknown')}")
        return {
            "status": "success",
            "data": result.data[0]
        }
    except Exception as e:
        print(f"Error in get_try_on_status for {try_on_id}: {str(e)}")
        # Return a safe response instead of an error to prevent frontend crashes
        return {
            "status": "success",
            "data": {
                "id": try_on_id,
                "status": "pending",
                "message": "Status check temporarily unavailable"
            }
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)