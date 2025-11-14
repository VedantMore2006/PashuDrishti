# app.py  -- Ready to replace your old app.py (based on your final_app.py logic)
import os
import warnings
import json
import hashlib
from io import BytesIO
from typing import Optional

import requests
from PIL import Image
from dotenv import load_dotenv

# Optional SDK import guarded so builds that don't have google-generativeai don't break.
try:
    import google.generativeai as genai  # type: ignore
    _HAS_GENAI_SDK = True
except Exception:
    genai = None  # type: ignore
    _HAS_GENAI_SDK = False

import torch
import timm
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

# ---------------------------
# App
# ---------------------------
app = FastAPI(
    title="Pashu Drishti Breed Identification API",
    description="Upload an image to classify bovine breed and optionally get Gemini-enriched hints.",
    version="2.0.0"
)

# Serve repo files if needed (optional)
app.mount("/static", StaticFiles(directory="."), name="static")

# CORS
ALLOW_WILDCARD_CORS = os.getenv("ALLOW_WILDCARD_CORS", "true").lower() in ("1", "true", "yes")
allowed_origins = [
    "http://localhost:7860",
    "http://127.0.0.1:7860",
    "http://localhost:8000",
]
if ALLOW_WILDCARD_CORS:
    allowed_origins.append("*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Model settings (same logic as your final_app.py)
# ---------------------------
# Default file names: prefer `best.pt` (you have it in repo), but keep compatibility with final file.
MODEL_PATH = os.getenv("MODEL_PATH", "best_model_final.pth")
MODEL_NAME = os.getenv("MODEL_ARCH", "convnext_tiny")
DROP_PATH_RATE = float(os.getenv("DROP_PATH_RATE", 0.2))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Breed metadata (kept from your final_app.py)
breed_metadata_details = {
    "Alambadi": {"family": "Cattle", "traits": "Draught breed from Tamil Nadu, grey or dark grey coat, backward-curving horns, well-built and hardy", "nutrition": "Primarily for work, thrives on local grazing; avg. milk yield 1-2 L/day"},
    "Amritmahal": {"family": "Cattle", "traits": "Grey coat, long tapering head, long horns emerging close together, known for power and endurance", "nutrition": "Primarily a draught animal, requires high energy feed for work; low milk yield avg. 1-2 L/day"},
    "Ayrshire": {"family": "Cattle", "traits": "Exotic dairy breed from Scotland, cherry red and white coat, small pointed horns", "nutrition": "Efficient grazer suited for butter and cheese production; avg. milk yield 18–22 L/day"},
    "Banni": {"family": "Buffalo", "traits": "Resilient breed from Kutch, Gujarat; typically black, can survive on sparse vegetation", "nutrition": "Adapted to arid conditions, known for high milk productivity; avg. milk yield 6–12 L/day"},
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
classes = sorted(list(breed_metadata_details.keys()))
NUM_CLASSES = len(classes)

# -------------------------------
# Load model
# -------------------------------
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, drop_path_rate=DROP_PATH_RATE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print(f"✅ PyTorch Model '{MODEL_NAME}' loaded successfully from {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Could not load PyTorch model: {e}") from e

# -------------------------------
# Gemini config (safe)
# -------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
# prefer GEMINI_MODEL env; fallback to gemini-2.5-pro (your allowed list likely contains this)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
_genai_client_configured = False
if _HAS_GENAI_SDK and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)  # type: ignore
        _genai_client_configured = True
        print("✅ google.generativeai SDK configured (optional).")
    except Exception as e:
        print("⚠️ google.generativeai SDK present but configuration failed:", e)

# -------------------------------
# Preprocessing (same as final_app)
# -------------------------------
preprocess_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes: bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        t = preprocess_transform(img)
        return t.unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

def sha256_of_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

# -------------------------------
# Gemini helper - best-effort (SDK preferred, REST fallback). Non-fatal.
# -------------------------------
def parse_gemini_text_response(resp_json):
    """
    Different SDK / REST versions return different shapes.
    Try a few common keys to extract the generated text.
    """
    try:
        # SDK style might have 'candidates' or 'output' or 'choices'
        if isinstance(resp_json, dict):
            # common REST candidate path
            if "candidates" in resp_json and len(resp_json["candidates"]) > 0:
                c = resp_json["candidates"][0]
                return c.get("content") or c.get("output") or c.get("text") or json.dumps(c)
            # Google newer API might have 'output' keys or 'results'
            if "output" in resp_json:
                # 'output' might be list of items with 'content'
                out = resp_json["output"]
                if isinstance(out, list) and len(out) > 0:
                    # first element content may be dict/list
                    first = out[0]
                    if isinstance(first, dict):
                        for k in ("content", "text", "type"):
                            if k in first:
                                return first.get(k)
                    return json.dumps(first)
            if "text" in resp_json:
                return resp_json["text"]
            if "response" in resp_json:
                return resp_json["response"]
        # fallback - stringify
        return json.dumps(resp_json)
    except Exception:
        return None

async def get_enhanced_details_from_gemini(breed_name: str, traits: str, nutrition: str):
    """
    Try SDK first (if available and configured). If that fails, attempt
    a REST call to the generative endpoint. If everything fails, return None.
    The app will fall back to static metadata in that case.
    """
    if not GEMINI_API_KEY:
        return None

    prompt = (
        f"You are an expert veterinarian. Return ONLY a valid JSON object (no commentary). "
        f"Schema: {{'ai_summary':str,'enhanced_traits':str,'improved_nutrition_plan':str,'management_tip':str}}. "
        f"Breed: {breed_name}. Traits: {traits}. Nutrition: {nutrition}. Keep it concise and farmer-friendly."
    )

    # 1) SDK path (best-effort)
    if _genai_client_configured and genai is not None:
        try:
            # The SDK surface varies by version; use a common helper if available.
            # Many older SDKs have a simple generate_text / GenerativeModel API.
            # Try multiple safe calls (wrapped in try/except).
            try:
                # new-ish SDK pattern: genai.generate with model parameter
                resp = genai.generate_text(model=GEMINI_MODEL, prompt=prompt, max_output_tokens=200)  # type: ignore
                # Parse response safely
                txt = None
                if isinstance(resp, dict):
                    txt = parse_gemini_text_response(resp)
                else:
                    # Some SDKs return an object with .text or .output
                    txt = getattr(resp, "text", None) or getattr(resp, "output", None) or str(resp)
                if txt:
                    # ensure valid JSON
                    try:
                        cleaned = txt.strip().lstrip("```").rstrip("```")
                        return json.loads(cleaned)
                    except Exception:
                        # sometimes model returns plain text; safe fallback: wrap fields
                        return {"ai_summary": txt[:200], "enhanced_traits": traits, "improved_nutrition_plan": nutrition, "management_tip": ""}
            except Exception:
                # Try alternate SDK surface: GenerativeModel usage
                try:
                    gm = genai.GenerativeModel(GEMINI_MODEL)  # type: ignore
                    resp = gm.generate(prompt)  # type: ignore
                    txt = parse_gemini_text_response(resp if isinstance(resp, dict) else resp.__dict__)
                    if txt:
                        try:
                            return json.loads(txt.strip().lstrip("```").rstrip("```"))
                        except Exception:
                            return {"ai_summary": txt[:200], "enhanced_traits": traits, "improved_nutrition_plan": nutrition, "management_tip": ""}
                except Exception:
                    pass
        except Exception:
            pass  # fall through to REST fallback

    # 2) REST fallback (best-effort)
    try:
        url = f"https://generative.googleapis.com/v1/models/{GEMINI_MODEL}:generate"
        headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
        body = {
            "prompt": [{"content": prompt}],
            "maxOutputTokens": 200
        }
        r = requests.post(url, headers=headers, json=body, timeout=10)
        if r.ok:
            data = r.json()
            txt = parse_gemini_text_response(data)
            if txt:
                try:
                    cleaned = txt.strip().lstrip("```").rstrip("```")
                    return json.loads(cleaned)
                except Exception:
                    return {"ai_summary": txt[:200], "enhanced_traits": traits, "improved_nutrition_plan": nutrition, "management_tip": ""}
        else:
            # non-200; return None so caller uses fallback
            return None
    except Exception:
        return None

# -------------------------------
# Debug helpers & endpoints
# -------------------------------
@app.get("/model_info")
def model_info():
    path = MODEL_PATH
    sha = sha256_of_file(path)
    size = None
    try:
        size = os.path.getsize(path)
    except Exception:
        size = None
    names = None
    try:
        # many timm models don't have .names; return our classes mapping
        names = {i: name for i, name in enumerate(classes)}
    except Exception:
        names = None
    return {"path": path, "size": size, "sha256": sha, "classes": names, "device": DEVICE}

def results_to_simple(output_tensor: torch.Tensor):
    """
    Convert logits/probabilities tensor to numpy 1D vector for a single image.
    Accepts a tensor of shape (1, num_classes) or (num_classes,).
    """
    try:
        probs = F.softmax(output_tensor, dim=1) if output_tensor.dim() == 2 else F.softmax(output_tensor.unsqueeze(0), dim=1)
        vec = probs.cpu().numpy()[0]
        return vec
    except Exception:
        # fallback: try numpy conversion
        try:
            arr = output_tensor.cpu().numpy()
            if arr.ndim == 2:
                arr = arr[0]
            # apply softmax manually
            e = np.exp(arr - np.max(arr))
            return (e / e.sum()).astype(float)
        except Exception:
            return None

@app.post("/debug_predict")
async def debug_predict(file: UploadFile = File(...)):
    # Accept image, run model with a low debug confidence and return raw vector
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Upload an image file.")
    image_bytes = await file.read()
    tensor = preprocess_image(image_bytes).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
    # out expected shape (1, num_classes) - ensure we handle both tensor/list
    try:
        if isinstance(out, (list, tuple)) and len(out) == 1:
            out = out[0]
    except Exception:
        pass
    vec = results_to_simple(out)
    return JSONResponse({"raw_output_vector_length": len(vec) if vec is not None else None, "raw_vector_top5": (
        [{"class_idx": int(i), "class_name": classes[int(i)], "prob": float(vec[int(i)])} for i in np.argsort(vec)[-5:][::-1]] if vec is not None else None
    ), "model_info": model_info()})

# -------------------------------
# Main predict endpoint (keeps original final_app behavior)
# -------------------------------
@app.post("/predict")
async def predict_breed(file: UploadFile = File(...)):
    """
    1) Image upload -> preprocess -> model tensor -> inference
    2) Softmax -> top-3 indices
    3) Optionally call Gemini to enrich
    4) Return JSON
    """
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Upload an image file.")

    try:
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes).to(DEVICE)
        with torch.no_grad():
            outputs = model(image_tensor)
            # adjust if model returns list
            if isinstance(outputs, (list, tuple)) and len(outputs) == 1:
                outputs = outputs[0]
            probs = results_to_simple(outputs)
            if probs is None:
                raise RuntimeError("Failed to convert model outputs to probability vector.")
        # top-3
        top_indices = np.argsort(probs)[-3:][::-1]
        main_idx = int(top_indices[0])
        main_name = classes[main_idx]
        main_conf = float(probs[main_idx] * 100)
        main_info = breed_metadata_details.get(main_name, {})

        enhanced = await get_enhanced_details_from_gemini(main_name, main_info.get("traits", ""), main_info.get("nutrition", ""))
        if not enhanced:
            enhanced = {
                "ai_summary": "AI enhancement is currently unavailable.",
                "enhanced_traits": main_info.get("traits", "N/A"),
                "improved_nutrition_plan": main_info.get("nutrition", "N/A"),
                "management_tip": "N/A"
            }

        response = {
            "predicted_breed": main_name,
            "confidence": f"{main_conf:.2f}%",
            "family": main_info.get("family", "N/A"),
            "details": enhanced,
            "top_3_matches": [
                {"breed": classes[int(i)], "confidence": f"{float(probs[int(i)] * 100):.2f}%"}
                for i in top_indices
            ]
        }
        return JSONResponse(response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# -------------------------------
# Health/root
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_path": MODEL_PATH}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
