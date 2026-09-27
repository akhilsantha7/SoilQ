from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Literal, Optional
from anthropic import AsyncAnthropic
from fastapi.responses import JSONResponse
import os
import base64
import json

# -----------------------------
# App + Anthropic Client
# -----------------------------
app = FastAPI(title="SoilQ GenAI Service")
client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Model selection: set in env to override. Defaults to Claude Sonnet 4.5 —
# check https://docs.claude.com/en/docs/about-claude/models for the latest
# model id and bump ANTHROPIC_MODEL in Render's environment if a newer one
# is available. One model handles both text advice and vision (no separate
# vision model needed, unlike the old OpenAI setup).
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

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
You are a professional irrigation advisor for farmers. Give a **smart advisory**, not raw data.
Respond in {lang}.

### Current Field Conditions
- Crop: {req.crop_name or "Unknown"}
- Soil Moisture: {req.soil_moisture or 0}%
- Soil Temperature: {req.soil_temp or 0}°C
- Soil pH: {req.soil_ph or 0}
- Irrigation needed: {"Yes" if (req.irrigation_needed or 0) == 1 else "No"}
- Hours until irrigation recommended: {req.time_to_irrigation or 0}

### 7-Day Weather Forecast
{forecast_text}

### Required output format (smart advisory, not moisture display)
- Do NOT just say "Soil moisture: X%" or "Time to irrigation: Y hours."
- Give **actionable advice** in this style:

**If irrigation IS needed:**
1. First line: specific recommendation with 💧, e.g. "💧 Irrigate tomorrow 6 AM for 40 minutes" or "💧 Irrigate today evening (5–6 PM) for 35 minutes". Use the "hours until irrigation" and forecast to pick a concrete **day**, **time** (e.g. early morning), and **duration in minutes** (suggest 30–45 min typically; adjust by crop and soil).
2. Next: weather-based adjustment in one short line, e.g. "Rain expected in 48 hours — reduce duration by 20%" or "No rain in next 5 days — you can irrigate at full duration."
3. Optional: one brief water-saving or risk tip.

**If irrigation is NOT needed:**
- One clear line, e.g. "💧 No irrigation needed now. Soil moisture is sufficient. Next check in 2–3 days."
- Optional: mention when rain is expected or when to irrigate next.

- Use complete sentences. Respond only in {lang}. Do not mention AI or predictions.
"""

    try:
        response = await client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )

        advice_text = response.content[0].text.strip()

        # Always return valid JSON
        return JSONResponse(content={"advice": advice_text})

    except Exception as e:
        return JSONResponse(content={"advice": f"Error generating advice: {str(e)}"})


# -----------------------------
# Disease Advice
# -----------------------------
# Forcing a tool call (instead of asking the model to "reply with only JSON")
# means the SDK hands back schema-validated, already-parsed data — there is
# no raw text to mis-parse if the model adds a stray sentence, which is what
# was producing "Unknown condition" / empty advice under the old prompt-only
# JSON approach.
DISEASE_TEXT_ADVICE_TOOL = {
    "name": "report_disease_advice",
    "description": "Report 5 farmer-facing advice points about a diagnosed crop disease.",
    "input_schema": {
        "type": "object",
        "properties": {
            "advice_points": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 5,
                "maxItems": 5,
                "description": (
                    "Exactly 5 strings, in this order, each starting with its heading: "
                    "'Disease Overview', 'Immediate Actions', 'Control Options', "
                    "'Weather Considerations', 'Prevention Tips'."
                ),
            }
        },
        "required": ["advice_points"],
    },
}


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
- Each advice point should be 1–2 sentences.
- Do NOT mention AI or predictions.
"""

    try:
        response = await client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=500,
            tools=[DISEASE_TEXT_ADVICE_TOOL],
            tool_choice={"type": "tool", "name": "report_disease_advice"},
            messages=[{"role": "user", "content": prompt}],
        )

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        pages = tool_use.input.get("advice_points") if tool_use else None

        if not pages or not isinstance(pages, list) or not all(isinstance(p, str) for p in pages):
            pages = ["No advice available for this section."] * 5
        elif len(pages) < 5:
            pages = pages + ["No advice available for this section."] * (5 - len(pages))

        return JSONResponse(content={"advice": pages})

    except Exception as e:
        return JSONResponse(content={"advice": [f"Error generating advice: {str(e)}"]})


# -----------------------------
# Disease + Nutrition from Image (Claude Vision)
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

VISION_PROMPT = """You are an expert plant pathologist. Look at THIS specific image and base your answer ONLY on what you see (lesions, spots, color, mold, rot, etc.). Different images must get different disease_type when they show different conditions.

Allowed disease_type (pick the ONE that best matches what you see):
{class_names}

Visual cues to distinguish:
- Healthy: no spots, lesions, or discoloration; normal green leaf color.
- Anthracnose: dark, sunken lesions; may show pink/orange spore masses in wet conditions.
- Powdery Mildew: white or gray powdery coating on leaf surface.
- Sun Blotch: irregular discolored or streaked blotches (more common on fruit than leaves).
- Cercospora Leaf Spot: small circular spots, often gray center with dark brown or purple margin.
- Root Rot: generalized yellowing, wilting, canopy thinning, or decline without distinct leaf lesions (roots not visible; infer only from visible plant stress).
- Scab: raised, corky, or rough scabby lesions.
- Algal Leaf Spot: greenish, orange, or rust-colored velvety/fuzzy circular spots.
- None detected: image is not a plant/leaf/crop or too blurry to determine.

Base disease_type and disease_confidence strictly on THIS image only. Then call the report_disease_analysis tool with your findings — always call it, even for a healthy or unclear photo, picking your best-guess disease_type rather than leaving it out.""".format(
    class_names=", ".join(f'"{x}"' for x in DISEASE_CLASS_NAMES)
)

# Forcing a tool call here (instead of asking the model to "reply with only
# JSON") means the SDK hands back schema-validated, already-parsed data —
# there is no raw JSON string to mis-parse if the model adds so much as one
# stray sentence, which is what was producing "Unknown condition" under the
# old prompt-only JSON approach.
DISEASE_ANALYSIS_TOOL = {
    "name": "report_disease_analysis",
    "description": "Report the plant disease diagnosis found in the analyzed image.",
    "input_schema": {
        "type": "object",
        "properties": {
            "disease_type": {
                "type": "string",
                "enum": DISEASE_CLASS_NAMES,
                "description": "The single best-matching disease classification for the image.",
            },
            "disease_confidence": {
                "type": "number",
                "description": "Confidence 0-100 (percentage).",
            },
            "nutrition_deficiency": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Nutrient deficiencies visible, e.g. ['Nitrogen', 'Iron']. Empty array if none visible.",
            },
            "severity": {
                "type": "string",
                "enum": ["Mild", "Moderate", "Severe", "Healthy"],
            },
            "treatment_summary": {
                "type": "string",
                "description": "1-2 sentences specific to this condition.",
            },
            "treatment_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 concrete actionable steps.",
            },
            "other_observations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Pests, multiple symptoms, growth stage, environmental stress, etc.",
            },
        },
        "required": ["disease_type", "disease_confidence"],
    },
}


@app.post("/genai/disease-from-image", response_model=DiseaseImageAnalysis)
async def disease_from_image(
    image: UploadFile = File(...),
    crop_name: Optional[str] = Form(None),
    language: Optional[str] = Form("english"),
):
    """Accept a plant/leaf image, send to Claude Vision for disease type, nutrition deficiency, and treatment advice."""
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

    lang_map = {"english": "English", "hindi": "Hindi", "telugu": "Telugu"}
    lang = lang_map.get((language or "english").lower(), "English")
    lang_instruction = (
        f" Write all human-readable text (treatment_summary, treatment_steps, other_observations) in {lang}."
        + (" Use the native script (Devanagari for Hindi, Telugu script for Telugu)." if lang != "English" else "")
    )
    crop_note = f" Optional context: crop is '{crop_name}'." if crop_name else ""
    user_content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64,
            },
        },
        {
            "type": "text",
            "text": VISION_PROMPT + lang_instruction + crop_note,
        },
    ]

    try:
        response = await client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=800,
            tools=[DISEASE_ANALYSIS_TOOL],
            tool_choice={"type": "tool", "name": "report_disease_analysis"},
            messages=[{"role": "user", "content": user_content}],
        )

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if not tool_use:
            raise ValueError("Model did not return a tool_use block")
        data = tool_use.input

        # Map to our response model (allow extra keys from API)
        return DiseaseImageAnalysis(
            disease_type=data.get("disease_type"),
            disease_confidence=data.get("disease_confidence"),
            nutrition_deficiency=data.get("nutrition_deficiency") or [],
            severity=data.get("severity"),
            treatment_summary=data.get("treatment_summary"),
            treatment_steps=data.get("treatment_steps") or [],
            other_observations=data.get("other_observations") or [],
            raw_advice=json.dumps(data),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision analysis failed: {str(e)}")
