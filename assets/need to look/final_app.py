import os
import warnings
import json
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import torch
import timm
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Suppress minor warnings
# Suppress non-critical UserWarning messages to keep logs clean during development/runtime.
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables from a .env file (if present). This keeps secrets and
# config out of source control. E.g., GOOGLE_API_KEY, ALLOW_WILDCARD_CORS, etc.
load_dotenv()

# ==============================================================================
# 1. APP INITIALIZATION
# ==============================================================================
# Initialize the FastAPI application. Title/description/version are used by
# the OpenAPI docs (available at /docs when the server is running).
app = FastAPI(
    title="Pashu Drishti Breed Identification API 🐄 (AI Enhanced)",
    description="Upload an image of a bovine to classify its breed and get AI-powered insights.",
    version="2.0.0 (PyTorch + Gemini)"
)

# Mount static files so the app can serve assets if needed
# Mount a static file directory so the app can serve images or other assets at
# /static. This is optional but convenient for small frontend assets or testing.
app.mount("/static", StaticFiles(directory="."), name="static")

# CORS configuration with option to disable wildcard via environment variable
# CORS (Cross-Origin Resource Sharing) configuration. By default the app
# whitelists a few local development origins. For convenience during local
# testing we allow a wildcard origin when ALLOW_WILDCARD_CORS is true. In
# production you should set ALLOW_WILDCARD_CORS to false and list only the
# trusted frontend origins.
ALLOW_WILDCARD_CORS = os.getenv("ALLOW_WILDCARD_CORS", "true").lower() in ("1", "true", "yes")
allowed_origins = [
    # Example local development hosts; keep or modify these for your environment
    "http://192.168.68.119:7860",
    "http://localhost:7860",
    "http://localhost:8000",
]
if ALLOW_WILDCARD_CORS:
    # WARNING: allowing '*' opens your API to requests from any origin.
    # Use for development only or when behind proper authentication proxies.
    allowed_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 2. MODEL AND METADATA SETUP
# ==============================================================================
# Model configuration: path to the saved PyTorch checkpoint and model type.
MODEL_PATH = "best_model_final.pth"
MODEL_NAME = "convnext_tiny"
DROP_PATH_RATE = 0.2

# Decide whether to use GPU (CUDA) or CPU based on availability. Using the
# GPU will greatly speed up inference when available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Breed metadata (unchanged)
# Breed metadata provides human-readable information for each class the
# classifier can predict. This is not used during model inference; it is
# returned to API users to provide helpful context and suggestions.
breed_metadata_details = {
    "Alambadi": {"family": "Cattle", "traits": "Draught breed from Tamil Nadu, grey or dark grey coat, backward-curving horns, well-built and hardy", "nutrition": "Primarily for work, thrives on local grazing; avg. milk yield 1-2 L/day"},
    "Amritmahal": {"family": "Cattle", "traits": "Grey coat, long tapering head, long horns emerging close together, known for power and endurance", "nutrition": "Primarily a draught animal, requires high energy feed for work; low milk yield avg. 1-2 L/day"},
    "Ayrshire": {"family": "Cattle", "traits": "Exotic dairy breed from Scotland, cherry red and white coat, small pointed horns", "nutrition": "Efficient grazer suited for butter and cheese production; avg. milk yield 18–22 L/day"},
    "Banni": {"family": "Buffalo", "traits": "Resilient breed from Kutch, Gujarat; typically black, can survive on sparse vegetation", "nutrition": "Adapted to arid conditions, known for high milk productivity; avg. milk yield 6-12 L/day"},
    "Bargur": {"family": "Cattle", "traits": "Draught breed from Tamil Nadu, brown with extensive white markings, known for being aggressive and semi-wild", "nutrition": "Excellent for work in hilly terrain, very low milk yield; avg. milk yield <1 L/day"},
    "Bhadawari": {"family": "Buffalo", "traits": "Copper-colored brownish coat, two white lines on the lower neck ('Chevron')", "nutrition": "Known for extremely high milk fat (6-14%); avg. milk yield 4–6 L/day"},
    "Brown_Swiss": {"family": "Cattle", "traits": "Exotic dairy breed from Switzerland, uniformly brown in color, large frame", "nutrition": "High-yield dairy animal, requires quality pasture; avg. milk yield 20–25 L/day"},
    "Dangi": {"family": "Cattle", "traits": "Draught breed from Maharashtra, irregular red/black and white patches, hardy for heavy rainfall areas", "nutrition": "Suited for draught work in rice fields, low milk yield; avg. milk yield 1-3 L/day"},
    "Deoni": {"family": "Cattle", "traits": "Spotted black and white coat, dropping ears, short blunt horns, prominent forehead", "nutrition": "Good dual-purpose animal from Marathwada region; avg. milk yield 4–6 L/day"},
    "Gir": {"family": "Cattle", "traits": "Red coat with white patches, broad convex forehead, long pendulous ears, prominent hump", "nutrition": "Hardy dairy breed, tolerates stress conditions; avg. milk yield 6–10 L/day"},
    "Guernsey": {"family": "Cattle", "traits": "Exotic dairy breed, fawn or red and white coat, produces golden-yellow tinted milk (high beta-carotene)", "nutrition": "Efficient milk producer on grass-based diets; avg. milk yield 15–20 L/day"},
    "Hallikar": {"family": "Cattle", "traits": "Premier draught breed, grey to dark grey, long coffin-shaped face, long pointed horns", "nutrition": "Excellent for draught work, requires high energy feed; low milk yield avg. 1–3 L/day"},
    "Hariana": {"family": "Cattle", "traits": "White or light grey coat, compact body, long face with a flat forehead, short horns", "nutrition": "Dual-purpose, powerful work animals, bullocks are prized; avg. milk yield 4–6 L/day"},
    "Holstein_Friesian": {"family": "Cattle", "traits": "Exotic breed from Holland, largest dairy breed, distinct black and white coat", "nutrition": "Highest milk producer in the world, requires high-quality feed; avg. milk yield 25–30 L/day"},
    "Jaffrabadi": {"family": "Buffalo", "traits": "Heaviest buffalo breed, black coat, prominent broad and convex forehead, drooping ring-like horns", "nutrition": "Males used for heavy draught work; avg. milk yield 7–10 L/day"},
    "Jersey": {"family": "Cattle", "traits": "Exotic breed from UK, smallest dairy breed, tan to cream color, dished forehead", "nutrition": "High milk fat content (5.4%), adaptable to various climates; avg. milk yield 15–20 L/day"},
    "Kangayam": {"family": "Cattle", "traits": "Premier draught breed, grey to dark grey coat, long pointed horns, calves are red at birth", "nutrition": "Hardy and thrives on local fodder, known for strength; avg. milk yield 2-4 L/day"},
    "Kankrej": {"family": "Cattle", "traits": "Heaviest Indian cattle breed, silver-grey coat, strong lyre-shaped horns, has a peculiar gait ('Sawai chal')", "nutrition": "Hardy and active, prized for both milk and draught; avg. milk yield 5–8 L/day"},
    "Kasargod": {"family": "Cattle", "traits": "Dwarf breed from Kerala, smaller than Vechur, known for climate resilience and disease resistance", "nutrition": "High milk yield for its body size, efficient converter of feed; avg. milk yield 2-3 L/day"},
    "Kenkatha": {"family": "Cattle", "traits": "Draught breed from Bundelkhand region, grey or black coat, short horns, very active and sturdy", "nutrition": "Excellent for light farming and transport in rugged terrain; low milk yield avg. 1-2 L/day"},
    "Kherigarh": {"family": "Cattle", "traits": "Draught breed from Uttar Pradesh, white coat, long face and large horns, known for its swiftness", "nutrition": "Active and fast-moving, used for light transport and ploughing; low milk yield avg. <1 L/day"},
    "Khillari": {"family": "Cattle", "traits": "Originated from Hallikar breed, greyish-white coat, long pointed horns, long face with bulged forehead", "nutrition": "Powerful and fast draught animal, adapted to drought areas; poor milk yielder avg. 1–3 L/day"},
    "Krishna_Valley": {"family": "Cattle", "traits": "Large, massive animals with a grey-white coat, long face and bulged forehead", "nutrition": "Developed for draught in black cotton soil areas; avg. milk yield 3-5 L/day"},
    "Malnad_gidda": {"family": "Cattle", "traits": "Dwarf breed from Karnataka, black or brown coat, very small and compact, disease resistant", "nutrition": "Suited to hilly, high-rainfall areas, milk known for medicinal properties; avg. milk yield 2-3 L/day"},
    "Mehsana": {"family": "Buffalo", "traits": "Cross between Murrah and Surti, black coat, sickle-shaped horns that are less curved than Murrah", "nutrition": "Persistent milker valued for ghee production; avg. milk yield 6–9 L/day"},
    "Murrah": {"family": "Buffalo", "traits": "Best buffalo breed in the world, jet black color, short and tightly curved spiral horns", "nutrition": "High-yield dairy buffalo, requires quality feed and care; avg. milk yield 7–12 L/day"},
    "Nagori": {"family": "Cattle", "traits": "Fine draught breed from Rajasthan, white coat, long and narrow face, known for trotting ability", "nutrition": "Famous for fast draught work, especially in light iron ploughs; avg. milk yield 2-3 L/day"},
    "Nagpuri": {"family": "Buffalo", "traits": "Also called Ellichpuri, black color, long flat horns that curve up to the shoulders", "nutrition": "Dual-purpose, males used for heavy but slow draught work; avg. milk yield 4–6 L/day"},
    "Nili_Ravi": {"family": "Buffalo", "traits": "Dairy breed from Punjab, black coat, often has white markings on face and legs (Panch Kalyani), wall eyes", "nutrition": "High-yielding dairy buffalo, comparable to Murrah; avg. milk yield 7-12 L/day"},
    "Nimari": {"family": "Cattle", "traits": "Draught breed from Madhya Pradesh, red coat with large white splashes, resembles Gir but is hardier", "nutrition": "Very sturdy and active, good for agricultural work; low milk yield avg. 2-3 L/day"},
    "Ongole": {"family": "Cattle", "traits": "Also known as Nellore cattle, large and heavy with a glossy-white coat, short stumpy horns", "nutrition": "Iconic dual-purpose breed, adaptable and hardy; avg. milk yield 4–7 L/day"},
    "Pulikulam": {"family": "Cattle", "traits": "Draught breed from Tamil Nadu, dark grey or black, known for being used in Jallikattu", "nutrition": "Very strong and aggressive, primarily used for sport and draught; very low milk yield avg. <1 L/day"},
    "Rathi": {"family": "Cattle", "traits": "Evolved from Sahiwal/Red Sindhi crosses, brown with white patches, medium-sized body", "nutrition": "Important dual-purpose breed from Rajasthan; avg. milk yield 6–10 L/day"},
    "Red_Dane": {"family": "Cattle", "traits": "Exotic dairy breed from Denmark, solid deep red color, heavy build", "nutrition": "High-yield dairy breed known for longevity and strong legs; avg. milk yield 20-25 L/day"},
    "Red_Sindhi": {"family": "Cattle", "traits": "Distinctly red coat, short stumpy horns, pendulous dewlap and sheath", "nutrition": "Hardy dairy breed, performs well on limited feed; avg. milk yield 7–10 L/day"},
    "Sahiwal": {"family": "Cattle", "traits": "Reddish-dun coat, loose skin, short stumpy horns, prominent pendulous dewlap", "nutrition": "Best indigenous dairy breed, requires balanced green fodder; avg. milk yield 8–12 L/day"},
    "Surti": {"family": "Buffalo", "traits": "Dairy breed from Gujarat, brown to silver-grey coat, sickle-shaped horns, two white neck bands ('chevron')", "nutrition": "Good dairy animal with high-fat milk, very economical feeder; avg. milk yield 5-8 L/day"},
    "Tharparkar": {"family": "Cattle", "traits": "From Thar desert, white or light grey coat, short thick horns, black switch (tail end)", "nutrition": "Excellent dual-purpose breed with high potential under good nutrition; avg. milk yield 6–10 L/day"},
    "Toda": {"family": "Buffalo", "traits": "Fawn to ash-grey coat, large heavy head, long and wide crescent-shaped horns", "nutrition": "Adapted to dense forests of Nilgiris, known for sweet-flavored milk; low milk yield avg. 2-3 L/day"},
    "Umblachery": {"family": "Cattle", "traits": "Grey with white markings on face and legs, small outwardly curved horns, known for being swift", "nutrition": "Excellent draught breed for wetlands and farms in Tamil Nadu; avg. milk yield 1-3 L/day"},
    "Vechur": {"family": "Cattle", "traits": "Smallest cattle breed, light red, black or fawn and white coat, small thin horns", "nutrition": "Produces large amount of milk for its small size and feed intake; avg. milk yield 2-3 L/day"}
}
# Prepare sorted class list and number of classes. Sorting ensures consistent
# ordering across runs; NUM_CLASSES is required when constructing the model.
classes = sorted(list(breed_metadata_details.keys()))
NUM_CLASSES = len(classes)

# -------------------------------
# Load the PyTorch model
# -------------------------------
try:
    # Validate the model file exists before attempting to load it.
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

    # Create the model architecture (not pretrained weights), matching the
    # number of classes used during training.
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, drop_path_rate=DROP_PATH_RATE)

    # Load checkpoint and extract state dict. Some checkpoints are saved as a
    # dict with a 'model_state_dict' key, others are plain state_dict objects.
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Load weights into the model and set evaluation mode.
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print(f"✅ PyTorch Model '{MODEL_NAME}' loaded successfully from {MODEL_PATH}")
except Exception as e:
    # Use exception chaining to preserve the original traceback for easier debugging.
    raise RuntimeError(f"Could not load PyTorch model: {e}") from e

# -------------------------------
# Configure Gemini (Generative AI)
# -------------------------------
try:
    # Read the API key from environment. It's intentionally required here so
    # the app can optionally provide richer AI-enhanced responses when set.
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    # Configure the google-generativeai client and create a model handle.
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Gemini AI model configured successfully.")
except Exception as e:
    # If Gemini is not configured (missing key or network issue) we keep
    # running but disable AI enhancements; the rest of the API remains usable.
    print(f"⚠️ WARNING: Gemini AI model could not be configured. AI-enhanced details will be disabled. Error: {e}")
    gemini_model = None

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
# Image preprocessing pipeline. Matches the transforms typically used during
# model training: resize to 224x224, convert to tensor, and normalize using
# ImageNet statistics (commonly used base normalization for transfer learning).
preprocess_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image(image_bytes):
    """Convert raw image bytes into a normalized PyTorch tensor batch.

    Steps:
    - Open image from bytes with PIL and convert to RGB (3 channels)
    - Apply the preprocessing pipeline defined above
    - Add a batch dimension (unsqueeze) since models expect batched inputs

    On failure this raises an HTTPException with a 400 status so the API
    returns a client-friendly error message.
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_tensor = preprocess_transform(img)
        # Unsqueeze to create shape (1, C, H, W) for a single image batch.
        return img_tensor.unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")


async def get_enhanced_details_from_gemini(breed_name: str, traits: str, nutrition: str):
    """Call Gemini to generate an enhanced, structured JSON for a breed.

    This function constructs a strict prompt that instructs the model to
    return only a valid JSON object following a fixed schema. This reduces
    parsing errors and makes downstream handling deterministic.

    If Gemini is not configured (gemini_model is None) the function returns
    None so the caller can use fallback data.
    """
    # Return immediately if the Gemini client is not available.
    if not gemini_model:
        return None

    # Strict prompt asking the model to emit only a JSON object following
    # the exact schema. This makes parsing safe as long as the model respects
    # the instruction (we still guard against JSON errors below).
    prompt = f"""
    You are an expert veterinarian and livestock management consultant. 
    A farmer has uploaded a photo of a bovine. The system has classified it as the breed "{breed_name}" 
    with the following baseline information:

    - Key Traits: "{traits}"
    - Nutrition Info: "{nutrition}"

    Your task:
    Return ONLY a **valid JSON object** with no commentary, code blocks, or explanations. 
    Follow exactly this schema:

    {{
    "ai_summary": "Write a 2–3 sentence, farmer-friendly summary of {breed_name}, 
    including its origin, primary use (dairy/draught/dual), and one standout characteristic.",
    "enhanced_traits": "Expand the Key Traits. 
    Explain why each trait matters (disease resistance, climate tolerance, productivity, temperament, etc.).",
    "improved_nutrition_plan": "Turn the nutrition info into a short actionable feeding plan. 
    Include typical fodder names (green fodder, dry fodder, concentrates), mineral mix, seasonal adjustments, and daily water needs.",
    "management_tip": "One highly practical, 
    actionable tip specific to {breed_name} regarding housing, health care, or handling. Use plain language."
    }}

    Rules:
    - Be concise but rich with information.
    - Avoid repetition of breed name in every sentence.
    - Use common Indian farming terms where possible (e.g., ‘napier grass’, ‘mustard cake’, ‘chaffed fodder’).
    - Keep output valid JSON only.
    """

    try:
        # Asynchronously request content from the Gemini model.
        response = await gemini_model.generate_content_async(prompt)

        # Clean code fences or surrounding markdown (sometimes models include them)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()

        # Parse the response as JSON and return the structured object.
        return json.loads(cleaned_response)
    except Exception as e:
        # If the call fails or returns invalid JSON, log and return None so the
        # API can fall back to non-AI details.
        print(f"Error calling Gemini API: {e}")
        return None

# ==============================================================================
# 4. API ENDPOINTS
# ==============================================================================

@app.get("/")
async def read_root():
    """Health-check endpoint. Returns a small JSON stating the API is running.

    Useful for load balancers, monitoring, or quick local checks.
    """
    return {"status": "ok", "message": "Bovine Breed Identification API is running."}


@app.post("/predict")
async def predict_breed(file: UploadFile = File(...)):
    """Primary prediction endpoint.

    Accepts a file upload (multipart/form-data). Steps:
    1. Read uploaded image bytes
    2. Preprocess into a tensor and move to DEVICE (CPU/GPU)
    3. Run a forward pass through the model inside torch.no_grad()
    4. Softmax the outputs to probabilities and pick the top-3 classes
    5. Optionally call Gemini to enrich the result with farmer-friendly advice
    6. Return a structured JSON response with predicted breed, confidence,
       family, detailed advice, and top-3 matches
    """
    try:
        # Read raw bytes from the uploaded file object asynchronously.
        image_bytes = await file.read()

        # Preprocess the raw image bytes into a batched tensor and send to DEVICE
        image_tensor = preprocess_image(image_bytes).to(DEVICE)

        # Run inference without computing gradients (faster & uses less memory)
        with torch.no_grad():
            outputs = model(image_tensor)
            # Convert logits to probabilities
            probabilities = F.softmax(outputs, dim=1)
        
        # Convert to a 1D numpy array (single example batch) for post-processing
        prediction_vector = probabilities.cpu().numpy()[0]

        # Get indices of top-3 predictions (sorted highest-first)
        top_indices = np.argsort(prediction_vector)[-3:][::-1]
        
        # Primary predicted class index and associated metadata
        main_breed_index = top_indices[0]
        main_breed_name = classes[main_breed_index]
        main_breed_confidence = float(prediction_vector[main_breed_index] * 100)
        main_breed_info = breed_metadata_details.get(main_breed_name, {})

        # --- AI-Powered Enhancement ---
        # Ask Gemini to provide farmer-friendly details following the strict JSON schema.
        enhanced_details = await get_enhanced_details_from_gemini(
            breed_name=main_breed_name,
            traits=main_breed_info.get("traits", "N/A"),
            nutrition=main_breed_info.get("nutrition", "N/A")
        )

        # Fallback if Gemini is not available or returned invalid output
        if not enhanced_details:
            enhanced_details = {
                "ai_summary": "AI enhancement is currently unavailable.",
                "enhanced_traits": main_breed_info.get("traits", "N/A"),
                "improved_nutrition_plan": main_breed_info.get("nutrition", "N/A"),
                "management_tip": "N/A"
            }
        # --- End AI Enhancement ---

        # Construct the JSON response expected by clients (frontend or API users)
        response = {
            "predicted_breed": main_breed_name,
            "confidence": f"{main_breed_confidence:.2f}%",
            "family": main_breed_info.get("family", "N/A"),
            "details": enhanced_details,
            "top_3_matches": [
                {
                    "breed": classes[i], 
                    "confidence": f"{float(prediction_vector[i] * 100):.2f}%"
                }
                for i in top_indices
            ]
        }
        return response
    except Exception as e:
        # Return a 500 error with a helpful message for debugging client-side.
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# ==============================================================================
# 5. SCRIPT EXECUTION
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("final_app:app", host="0.0.0.0", port=7860)
