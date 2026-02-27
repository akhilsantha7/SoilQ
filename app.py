from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Literal, Optional
from openai import OpenAI
from fastapi.responses import JSONResponse
import os
import base64
import json

# -----------------------------
# App + OpenAI Client
# -----------------------------
app = FastAPI(title="SoilQ GenAI Service")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model selection: set in env to override (e.g. OPENAI_MODEL=gpt-4o-mini for lower cost)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

# -----------------------------
# Models
# -----------------------------
class DailyForecast(BaseModel):
    date: str
    temp: float
    humidity: float
    wind: float
    condition: str


class AdviceRequest(BaseModel):
    advice_type: Literal["irrigation", "disease", "warmup"]

    # ---- irrigation ----
    irrigation_needed: Optional[float] = None
    irrigation_confidence: Optional[float] = None
    time_to_irrigation: Optional[float] = None
    soil_moisture: Optional[float] = None
    soil_temp: Optional[float] = None
    soil_ph: Optional[float] = None

    # ---- disease ----
    crop_name: Optional[str] = None
    disease_name: Optional[str] = None
    disease_confidence: Optional[float] = None

    # ---- shared ----
    forecast: List[DailyForecast] = []
    language: Optional[str] = "english"


class AdviceResponse(BaseModel):
    advice: str


# ---- Image-based disease / nutrition analysis ----
class DiseaseImageAnalysis(BaseModel):
    disease_type: Optional[str] = None
    disease_confidence: Optional[float] = None
    nutrition_deficiency: Optional[List[str]] = None
    severity: Optional[str] = None
    treatment_summary: Optional[str] = None
    treatment_steps: Optional[List[str]] = None
    other_observations: Optional[List[str]] = None
    raw_advice: Optional[str] = None


# -----------------------------
# Root (Health Check)
# -----------------------------
@app.get("/")
def health():
    return {"status": "SoilQ GenAI is running 🌱"}


# -----------------------------
# Main API
# -----------------------------
@app.post("/genai", response_model=AdviceResponse)
async def generate_advice(req: AdviceRequest):
    if req.advice_type == "warmup":
        # simple warmup response
        return AdviceResponse(advice="Warmup done ✅")

    if req.advice_type == "irrigation":
        return await irrigation_advice(req)

    if req.advice_type == "disease":
        return await disease_advice(req)

    raise HTTPException(status_code=400, detail="Invalid advice_type")


# -----------------------------
# Irrigation Advice
# -----------------------------
async def irrigation_advice(req: AdviceRequest):
    # Format forecast nicely
    forecast_text = "\n".join(
        f"- {d.date}: {d.temp:.1f}°C, {d.humidity:.0f}% humidity, "
        f"{d.wind:.1f} m/s wind, {d.condition}"
        for d in req.forecast
    ) or "No forecast available"

    # Use language-friendly phrases
    lang_map = {
        "english": "English",
        "hindi": "Hindi",
        "telugu": "Telugu"
    }
    lang = lang_map.get(req.language.lower(), "English")

    prompt = f"""
You are a professional irrigation advisor for farmers.
Respond in {lang}.

### Current Field Conditions
- Crop: {req.crop_name or "Unknown"}
- Soil Moisture: {req.soil_moisture or 0}%
- Soil Temperature: {req.soil_temp or 0}°C
- Soil pH: {req.soil_ph or 0}
- Irrigation Needed: {"YES" if (req.irrigation_needed or 0) == 1 else "NO"}
- Time to Irrigation: {req.time_to_irrigation or 0} hours

### 7-Day Weather Forecast
{forecast_text}

### Instructions
- Provide **complete, well-structured sentences** suitable for farmers.
- Example: "At this time, there is no need for irrigation because the soil moisture is sufficient."
- Mention exact timing if irrigation is needed: e.g., "Irrigate in 2 days" or "within 6 hours."
- Explain reason using forecast (rain, temperature, humidity, wind).
- Provide water-saving tips and risk warnings.
- Avoid single-word answers.
- Respond only in the requested language.
- DO NOT mention AI, ML, or predictions.
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )

        advice_text = response.choices[0].message.content.strip()

        # Always return valid JSON
        return JSONResponse(content={"advice": advice_text})

    except Exception as e:
        return JSONResponse(content={"advice": f"Error generating advice: {str(e)}"})


# -----------------------------
# Disease Advice
# -----------------------------
async def disease_advice(req: AdviceRequest):
    # Format 7-day forecast
    forecast_text = "\n".join(
        f"- {d.date}: {d.temp:.1f}°C, {d.humidity:.0f}% humidity, "
        f"{d.wind:.1f} m/s wind, {d.condition}"
        for d in req.forecast
    ) or "No forecast available"

    # Map language
    lang_map = {
        "english": "English",
        "hindi": "Hindi",
        "telugu": "Telugu"
    }
    lang = lang_map.get(req.language.lower(), "English")

    # Prompt for the AI
    prompt = f"""
You are a professional plant pathologist.
Respond in {lang}.

### Crop & Disease Info
- Crop: {req.crop_name or "Unknown"}
- Detected Disease: {req.disease_name or "Unknown"}
- Confidence: {int((req.disease_confidence or 0) * 100)}%

### 7-Day Weather Forecast
{forecast_text}

### Instructions
- Provide 5 concise advice points for farmers, each with a heading:
    Disease Overview
    Immediate Actions
    Control Options
    Weather Considerations
    Prevention Tips
- Return a **JSON array** of 5 strings, each string containing the heading + advice.
- Each advice point should be 1–2 sentences.
- Do NOT include AI mentions or extra text outside the JSON array.
"""

    try:
        # Call OpenAI
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        advice_text = response.choices[0].message.content.strip()

        # Attempt to parse JSON array
        pages = []
        try:
            pages = json.loads(advice_text)
            # Validate: must be list of strings
            if not isinstance(pages, list) or not all(isinstance(p, str) for p in pages):
                raise ValueError("Not a valid JSON array of strings")
        except Exception:
            # If parsing fails, split by headings as fallback
            headings = [
                "Disease Overview",
                "Immediate Actions",
                "Control Options",
                "Weather Considerations",
                "Prevention Tips"
            ]
            for h in headings:
                if h in advice_text:
                    start = advice_text.find(h)
                    # Find next heading
                    next_starts = [advice_text.find(nh) for nh in headings if advice_text.find(nh) > start]
                    end = min(next_starts) if next_starts else len(advice_text)
                    pages.append(advice_text[start:end].strip())
            # Ensure 5 elements
            while len(pages) < 5:
                pages.append("No advice available for this section.")

        return JSONResponse(content={"advice": pages})

    except Exception as e:
        return JSONResponse(content={"advice": [f"Error generating advice: {str(e)}"]})


# -----------------------------
# Disease + Nutrition from Image (OpenAI Vision)
# -----------------------------
# Allowed disease classes for detection (use exactly these labels)
DISEASE_CLASS_NAMES = [
  "Healthy",
  "Anthracnose",
  "Powdery Mildew",
  "Sun Blotch",
  "Cercospora Leaf Spot",
  "Root Rot",
  "Scab",
  "Algal Leaf Spot"
]

VISION_PROMPT = """You are an expert plant pathologist. Look at THIS specific image and base your answer ONLY on what you see (lesions, spots, color, mold, rot, etc.). Each image must get ONE disease_type corresponding to the most dominant visible condition. Do NOT guess.

Allowed disease_type (pick the ONE that best matches what you see):
"Healthy", "Anthracnose", "Powdery Mildew", "Sun Blotch", "Cercospora Leaf Spot", "Root Rot", "Scab", "Algal Leaf Spot"

Visual cues to distinguish:
- Healthy: no spots, lesions, or discoloration; normal green leaf color.
- Anthracnose: dark, sunken lesions; may show pink/orange spore masses in wet conditions.
- Powdery Mildew: white or gray powdery coating on leaf surface.
- Sun Blotch: irregular discolored or streaked blotches (more common on fruit than leaves).
- Cercospora Leaf Spot: small circular spots, often gray center with dark brown or purple margin.
- Root Rot: generalized yellowing, wilting, or canopy decline; roots not visible; choose Healthy if leaves appear normal.
- Scab: raised, corky, or rough scabby lesions.
- Algal Leaf Spot: greenish, orange, or rust-colored velvety/fuzzy circular spots.

Only include nutrition_deficiency if clear visual signs are visible. Currently, only detect:
- Nitrogen deficiency: uniform yellowing of older leaves; do NOT guess other nutrients.

Rules:
- If disease_type is Healthy, severity MUST be "Healthy".
- Severity: "Mild", "Moderate", "Severe", or "Healthy".
- Select the most dominant disease if multiple symptoms appear.
- disease_confidence: give as a percentage (0–100), where 100 = very certain.
- Return a single JSON object with NO extra text.

Return JSON object with EXACT keys:
- disease_type: string
- disease_confidence: number (0–100)
- nutrition_deficiency: array of strings or []
- severity: string or null
- treatment_summary: string or null
- treatment_steps: array of strings or null
- other_observations: array of strings or null

Base disease_type and disease_confidence strictly on THIS image only. Respond with ONLY the JSON object.""".format(
    class_names=", ".join(f'"{x}"' for x in DISEASE_CLASS_NAMES)
)


@app.post("/genai/disease-from-image", response_model=DiseaseImageAnalysis)
async def disease_from_image(
    image: UploadFile = File(...),
    crop_name: Optional[str] = Form(None),
    language: Optional[str] = Form("english"),
):
    """Accept a plant/leaf image, send to OpenAI Vision for disease type, nutrition deficiency, and treatment advice."""
    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    if image.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed)}",
        )

    content = await image.read()
    if len(content) > 10 * 1024 * 1024:  # 10 MB
        raise HTTPException(status_code=400, detail="Image too large (max 10 MB)")

    b64 = base64.standard_b64encode(content).decode("utf-8")
    media_type = image.content_type or "image/jpeg"

    lang_note = f" Optional context: crop is '{crop_name}'." if crop_name else ""
    user_content = [
        {
            "type": "text",
            "text": VISION_PROMPT + lang_note + "\nRespond with ONLY the JSON object, no other text.",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"},
        },
    ]

    try:
        response = client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code block if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        data = json.loads(raw)

        # Map to our response model (allow extra keys from API)
        return DiseaseImageAnalysis(
            disease_type=data.get("disease_type"),
            disease_confidence=data.get("disease_confidence"),
            nutrition_deficiency=data.get("nutrition_deficiency") or [],
            severity=data.get("severity"),
            treatment_summary=data.get("treatment_summary"),
            treatment_steps=data.get("treatment_steps") or [],
            other_observations=data.get("other_observations") or [],
            raw_advice=raw,
        )
    except json.JSONDecodeError as e:
        return DiseaseImageAnalysis(
            other_observations=[f"Analysis completed but response was not valid JSON: {e}"],
            raw_advice=raw,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision analysis failed: {str(e)}")