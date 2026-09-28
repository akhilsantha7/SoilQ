from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Header
from pydantic import BaseModel
from typing import List, Literal, Optional
from anthropic import AsyncAnthropic
from fastapi.responses import JSONResponse
from datetime import date
import os
import base64
import json
import requests

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


# ---- Image-based fruit ripeness analysis ----
class RipenessImageAnalysis(BaseModel):
    ripeness_stage: Optional[str] = None
    ripeness_confidence: Optional[float] = None
    harvest_advice: Optional[str] = None
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
# Per-crop disease class lists + visual cues. Keyed by lowercased crop name
# (matches the `crop_name` form field the iOS app sends). Adding a new crop
# means adding one entry here — see build_vision_prompt/build_disease_tool
# below, which pick the right list per request instead of using one global
# Avocado-only list for every crop.
#
# A crop typed by the farmer that ISN'T in this dict (the "type your own"
# option on the plant picker) is not silently mapped to one of these lists —
# that would misdiagnose e.g. a tomato photo using avocado's disease classes.
# Instead it falls through to the free-text path below: the model still
# names a specific disease, just without a fixed enum to pick from, and the
# prompt says so explicitly so it doesn't invent visual cues for a crop it
# has no curated list for.
CROP_DISEASE_INFO = {
    "avocado": {
        "classes": [
            "Healthy",
            "Anthracnose",
            "Powdery Mildew",
            "Sun Blotch",
            "Cercospora Leaf Spot",
            "Root Rot",
            "Scab",
            "Algal Leaf Spot",
        ],
        "visual_cues": """- Healthy: no spots, lesions, or discoloration; normal green leaf color.
- Anthracnose: dark, sunken lesions; may show pink/orange spore masses in wet conditions.
- Powdery Mildew: white or gray powdery coating on leaf surface.
- Sun Blotch: irregular discolored or streaked blotches (more common on fruit than leaves).
- Cercospora Leaf Spot: small circular spots, often gray center with dark brown or purple margin.
- Root Rot: generalized yellowing, wilting, canopy thinning, or decline without distinct leaf lesions (roots not visible; infer only from visible plant stress).
- Scab: raised, corky, or rough scabby lesions.
- Algal Leaf Spot: greenish, orange, or rust-colored velvety/fuzzy circular spots.""",
    },
    "paddy": {
        "classes": [
            "Healthy",
            "Bacterial Leaf Blight",
            "Rice Blast",
            "Brown Spot",
            "Leaf Smut",
            "Sheath Blight",
            "Bacterial Leaf Streak",
            "Tungro",
        ],
        "visual_cues": """- Healthy: uniform green color, no lesions, spots, or discoloration.
- Bacterial Leaf Blight: water-soaked yellow-to-white lesions with wavy margins, usually starting at the leaf tip or edges and spreading downward; lesions dry to grayish-white.
- Rice Blast: diamond or spindle-shaped lesions with gray-white centers and reddish-brown to dark-brown borders, scattered across the leaf blade.
- Brown Spot: small, round to oval brown spots with a yellow halo, scattered evenly across the leaf.
- Leaf Smut: tiny, angular black spots scattered on the upper leaf surface, often clustered near the leaf tip.
- Sheath Blight: irregular greenish-gray blotches with brown margins on the leaf sheath, usually starting near the waterline and moving upward.
- Bacterial Leaf Streak: narrow, dark-green, water-soaked interveinal streaks that turn yellowish-brown and look translucent when held up to light.
- Tungro: yellow-to-orange leaf discoloration combined with stunted, bushy plant growth (viral, spread by leafhoppers).""",
    },
}


def build_vision_prompt(display_crop: str, info: Optional[dict]) -> str:
    if info:
        return """You are an expert plant pathologist examining a {crop} plant/leaf. Look at THIS specific image and base your answer ONLY on what you see (lesions, spots, color, mold, rot, etc.). Different images must get different disease_type when they show different conditions.

Allowed disease_type (pick the ONE that best matches what you see):
{class_names}

Visual cues to distinguish:
{visual_cues}
- None detected: image is not a plant/leaf/crop or too blurry to determine.

Base disease_type and disease_confidence strictly on THIS image only. Then call the report_disease_analysis tool with your findings — always call it, even for a healthy or unclear photo, picking your best-guess disease_type rather than leaving it out.""".format(
            crop=display_crop,
            class_names=", ".join(f'"{x}"' for x in info["classes"]),
            visual_cues=info["visual_cues"],
        )

    # No curated disease list for this crop (farmer typed a crop name outside
    # the picker) — still tell the model which plant it is (that alone rules
    # out most misidentification risk), but let it name the disease itself
    # rather than forcing a fit into an unrelated crop's class list.
    return f"""You are an expert plant pathologist examining a {display_crop} plant/leaf photo. Base your answer ONLY on what you see in THIS image (lesions, spots, discoloration, mold, rot, wilting, pest damage, etc.) — do not guess from general knowledge of {display_crop} beyond what's visible here.

There is no fixed disease list configured for {display_crop} in this system, so use your own plant-pathology knowledge and name the specific disease, pest, or nutrient deficiency you observe, in plain English (e.g. "Early Blight", "Aphid Infestation", "Nitrogen Deficiency"). If the plant looks healthy, set disease_type to "Healthy". If the image doesn't clearly show a plant/leaf, set it to "Unable to determine".

Then call the report_disease_analysis tool with your findings — always call it, even for a healthy or unclear photo, picking your best-guess rather than leaving it out."""


# Forcing a tool call here (instead of asking the model to "reply with only
# JSON") means the SDK hands back schema-validated, already-parsed data —
# there is no raw JSON string to mis-parse if the model adds so much as one
# stray sentence, which is what was producing "Unknown condition" under the
# old prompt-only JSON approach. The tool's disease_type enum is built per
# crop so the model can only pick from that crop's actual disease list; for
# an uncurated crop, disease_type falls back to a free-text field instead of
# an enum (still forced through the tool call, so still schema-validated —
# just without a fixed set of allowed values).
def build_disease_tool(info: Optional[dict]) -> dict:
    disease_type_schema = (
        {
            "type": "string",
            "enum": info["classes"],
            "description": "The single best-matching disease classification for the image.",
        }
        if info
        else {
            "type": "string",
            "description": "The plant disease, pest, or deficiency name in plain English, or 'Healthy' / 'Unable to determine'.",
        }
    )
    return {
        "name": "report_disease_analysis",
        "description": "Report the plant disease diagnosis found in the analyzed image.",
        "input_schema": {
            "type": "object",
            "properties": {
                "disease_type": disease_type_schema,
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

    crop_name_raw = (crop_name or "").strip()
    crop_key = crop_name_raw.lower()
    info = CROP_DISEASE_INFO.get(crop_key)  # None if this crop has no curated list
    display_crop = crop_name_raw.capitalize() if crop_name_raw else "plant"
    vision_prompt = build_vision_prompt(display_crop, info)
    disease_tool = build_disease_tool(info)

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
            "text": vision_prompt + lang_instruction,
        },
    ]

    try:
        response = await client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=800,
            tools=[disease_tool],
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


# -----------------------------
# Fruit Ripeness from Image (Claude Vision)
# -----------------------------
# Same crop-aware pattern as CROP_DISEASE_INFO above: curated fruits get a
# fixed ripeness scale + visual cues for a reliable read; anything typed
# outside this list (the "Checking something else?" field on the fruit
# picker) falls through to a free-text ripeness assessment instead of being
# wrongly matched against one of these scales.
CROP_RIPENESS_INFO = {
    "mango": {
        "stages": ["Unripe", "Ripe", "Overripe", "Spoiled"],
        "visual_cues": """- Unripe: Skin mostly green, very firm, tight smooth surface, no fragrance.
- Ripe: Skin has turned yellow/orange/red-blush (variety dependent), slight give when pressed gently, sweet fragrance near the stem.
- Overripe: Skin very soft and wrinkled, dark orange-to-brown patches, strong sweet smell, slight shriveling.
- Spoiled: Visible mold, black or sunken patches, leaking juice, fermented smell, insect damage.""",
    },
    "banana": {
        "stages": ["Unripe", "Ripe", "Overripe", "Spoiled"],
        "visual_cues": """- Unripe: Skin fully green, firm, no brown spots.
- Ripe: Skin yellow with a few small brown "sugar spots", firm but yields slightly to gentle pressure.
- Overripe: Skin mostly brown or black, flesh visibly soft/mushy through the skin, strong sweet smell.
- Spoiled: Skin fully black, split or leaking, visible mold, fermented odor.""",
    },
    "papaya": {
        "stages": ["Unripe", "Ripe", "Overripe", "Spoiled"],
        "visual_cues": """- Unripe: Skin fully green, very firm, no color break.
- Ripe: Skin mostly yellow-orange and fairly uniform, slight give when pressed.
- Overripe: Skin heavily wrinkled and very soft, dark orange-to-brown blotches.
- Spoiled: Moldy patches, sunken soft spots, leaking, fermented odor.""",
    },
    "tomato": {
        "stages": ["Unripe", "Ripe", "Overripe", "Spoiled"],
        "visual_cues": """- Unripe: Skin green or green with slight blush, very firm.
- Ripe: Skin fully red/orange (variety dependent), glossy, firm but yields slightly to gentle pressure.
- Overripe: Skin very soft, wrinkled or cracked, deep red color, may show splitting.
- Spoiled: Visible mold, sunken watery patches, collapsed structure, white/gray mold or black rot spots.""",
    },
}


def build_ripeness_prompt(display_fruit: str, info: Optional[dict]) -> str:
    if info:
        return """You are an expert postharvest specialist examining a {fruit} photo. Look at THIS specific image and base your answer ONLY on what you see (skin color, firmness cues inferred from texture/wrinkling, blemishes, mold, softness, etc.). Different images must get different ripeness_stage when they show different conditions.

Allowed ripeness_stage (pick the ONE that best matches what you see):
{stage_names}

Visual cues to distinguish:
{visual_cues}
- Unable to determine: image is not a {fruit}/fruit or too blurry to determine.

Base ripeness_stage and ripeness_confidence strictly on THIS image only. Then call the report_ripeness_analysis tool with your findings — always call it, even for a clearly spoiled or unclear photo, picking your best-guess ripeness_stage rather than leaving it out.""".format(
            fruit=display_fruit,
            stage_names=", ".join(f'"{x}"' for x in info["stages"]),
            visual_cues=info["visual_cues"],
        )

    # No curated ripeness scale for this fruit (farmer typed a fruit name
    # outside the picker) — still tell the model which fruit it is, but let
    # it describe the ripeness stage itself rather than forcing a fit into
    # an unrelated fruit's scale.
    return f"""You are an expert postharvest specialist examining a {display_fruit} photo. Base your answer ONLY on what you see in THIS image (skin color, firmness cues inferred from texture/wrinkling, blemishes, mold, softness, etc.) — do not guess from general knowledge of {display_fruit} beyond what's visible here.

There is no fixed ripeness scale configured for {display_fruit} in this system, so use your own postharvest knowledge and describe the ripeness stage in plain English (e.g. "Unripe", "Ripe", "Overripe", "Spoiled", or a more specific stage if that fits the fruit better). If the image doesn't clearly show a fruit, set ripeness_stage to "Unable to determine".

Then call the report_ripeness_analysis tool with your findings — always call it, even for a spoiled or unclear photo, picking your best-guess rather than leaving it out."""


def build_ripeness_tool(info: Optional[dict]) -> dict:
    ripeness_stage_schema = (
        {
            "type": "string",
            "enum": info["stages"],
            "description": "The single best-matching ripeness stage for the image.",
        }
        if info
        else {
            "type": "string",
            "description": "The ripeness stage in plain English, or 'Unable to determine'.",
        }
    )
    return {
        "name": "report_ripeness_analysis",
        "description": "Report the fruit ripeness assessment found in the analyzed image.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ripeness_stage": ripeness_stage_schema,
                "ripeness_confidence": {
                    "type": "number",
                    "description": "Confidence 0-100 (percentage).",
                },
                "harvest_advice": {
                    "type": "string",
                    "description": (
                        "1-2 sentence actionable recommendation for harvest timing or sale, "
                        "specific to this ripeness stage, e.g. 'Harvest within 2-3 days for best "
                        "market price.' or 'Sell immediately — past peak ripeness.'"
                    ),
                },
                "other_observations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Blemishes, pest damage, uneven ripening, bruising, etc.",
                },
            },
            "required": ["ripeness_stage", "ripeness_confidence"],
        },
    }


@app.post("/genai/ripeness-from-image", response_model=RipenessImageAnalysis)
async def ripeness_from_image(
    image: UploadFile = File(...),
    fruit_name: Optional[str] = Form(None),
    language: Optional[str] = Form("english"),
):
    """Accept a fruit image, send to Claude Vision for a ripeness stage and harvest/sale advice."""
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
        f" Write all human-readable text (harvest_advice, other_observations) in {lang}."
        + (" Use the native script (Devanagari for Hindi, Telugu script for Telugu)." if lang != "English" else "")
    )

    fruit_name_raw = (fruit_name or "").strip()
    fruit_key = fruit_name_raw.lower()
    info = CROP_RIPENESS_INFO.get(fruit_key)  # None if this fruit has no curated scale
    display_fruit = fruit_name_raw.capitalize() if fruit_name_raw else "fruit"
    ripeness_prompt = build_ripeness_prompt(display_fruit, info)
    ripeness_tool = build_ripeness_tool(info)

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
            "text": ripeness_prompt + lang_instruction,
        },
    ]

    try:
        response = await client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=600,
            tools=[ripeness_tool],
            tool_choice={"type": "tool", "name": "report_ripeness_analysis"},
            messages=[{"role": "user", "content": user_content}],
        )

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if not tool_use:
            raise ValueError("Model did not return a tool_use block")
        data = tool_use.input

        return RipenessImageAnalysis(
            ripeness_stage=data.get("ripeness_stage"),
            ripeness_confidence=data.get("ripeness_confidence"),
            harvest_advice=data.get("harvest_advice"),
            other_observations=data.get("other_observations") or [],
            raw_advice=json.dumps(data),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ripeness analysis failed: {str(e)}")


# -----------------------------
# Weather Alerts (push notifications via Firebase Cloud Messaging)
# -----------------------------
# Triggered by an external scheduler (e.g. a free cron-job.org ping or a
# GitHub Actions scheduled workflow) hitting POST /genai/check-weather-alerts
# every few hours — deliberately NOT an in-process scheduler, since Render's
# free tier spins the service down when idle and would silently stop firing
# if the check lived inside this process instead.
#
# Requires two env vars on Render, set once:
#   FIREBASE_SERVICE_ACCOUNT_JSON — the full JSON key from Firebase Console
#     > Project Settings > Service Accounts > Generate new private key.
#   ALERT_JOB_SECRET              — any random string; the caller must send
#     it back as the X-Alert-Secret header, so this endpoint can't be
#     triggered by anyone who finds the URL.
import firebase_admin
from firebase_admin import credentials, firestore as admin_firestore, messaging

_firebase_admin_app = None


def _ensure_firebase_admin():
    global _firebase_admin_app
    if _firebase_admin_app is not None:
        return _firebase_admin_app
    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON is not set")
    cred = credentials.Certificate(json.loads(raw))
    _firebase_admin_app = firebase_admin.initialize_app(cred)
    return _firebase_admin_app


def _evaluate_alert(day: dict) -> Optional[dict]:
    """`day` is one entry from _forecast_today_tomorrow. Checked most-to-least
    severe — only the first match is returned, so a day with both a heatwave
    and gusty wind gets one notification, not two."""
    if day["temp_max"] >= 40:
        return {
            "key": f"{day['date']}:heat",
            "title": "Extreme heat expected",
            "body": f"Up to {day['temp_max']:.0f}°C on {day['date']}. Irrigate early morning or evening.",
        }
    if day["precip_prob"] >= 70:
        return {
            "key": f"{day['date']}:rain",
            "title": "Heavy rain expected",
            "body": f"{day['precip_prob']:.0f}% chance of rain on {day['date']}. Plan irrigation and harvest around it.",
        }
    if day["wind_max"] >= 40:
        return {
            "key": f"{day['date']}:wind",
            "title": "High winds expected",
            "body": f"Gusts up to {day['wind_max']:.0f} km/h on {day['date']}. Secure young plants and stakes.",
        }
    if day["temp_min"] <= 4:
        return {
            "key": f"{day['date']}:cold",
            "title": "Cold snap expected",
            "body": f"As low as {day['temp_min']:.0f}°C on {day['date']}. Protect sensitive crops overnight.",
        }
    return None


def _forecast_today_tomorrow(lat: float, lon: float) -> List[dict]:
    """Same Open-Meteo endpoint the iOS app calls (WeatherViewModel.swift) —
    no API key needed, so the backend can reuse it directly."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max"
        "&timezone=auto&forecast_days=2"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    daily = resp.json()["daily"]
    return [
        {
            "date": daily["time"][i],
            "temp_max": daily["temperature_2m_max"][i],
            "temp_min": daily["temperature_2m_min"][i],
            "precip_prob": daily["precipitation_probability_max"][i],
            "wind_max": daily["wind_speed_10m_max"][i],
        }
        for i in range(len(daily["time"]))
    ]


@app.post("/genai/check-weather-alerts")
async def check_weather_alerts(x_alert_secret: Optional[str] = Header(None)):
    """Scans users with weatherAlertsEnabled == true, checks their forecast
    (via the lastLat/lastLon WeatherViewModel.swift saves), and sends one FCM
    push for the most severe condition found — at most once per unique
    (date, condition) per user, tracked via lastAlertKey on their user doc so
    calling this more often than needed doesn't spam anyone. Call it from an
    external scheduler (cron-job.org, GitHub Actions, etc.) every few hours."""
    expected_secret = os.getenv("ALERT_JOB_SECRET")
    if not expected_secret or x_alert_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Alert-Secret")

    try:
        _ensure_firebase_admin()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Firebase admin not configured: {str(e)}")

    db = admin_firestore.client()
    users_ref = db.collection("users").where("weatherAlertsEnabled", "==", True)

    checked = 0
    alerted = 0
    errors: List[str] = []

    for doc in users_ref.stream():
        checked += 1
        data = doc.to_dict() or {}
        token = data.get("fcmToken")
        lat = data.get("lastLat")
        lon = data.get("lastLon")
        if not token or lat is None or lon is None:
            continue

        try:
            days = _forecast_today_tomorrow(lat, lon)
        except Exception as e:
            errors.append(f"{doc.id}: forecast failed ({str(e)})")
            continue

        alert = None
        for day in days:
            alert = _evaluate_alert(day)
            if alert:
                break

        if not alert or data.get("lastAlertKey") == alert["key"]:
            continue  # nothing to report, or already sent this exact alert

        try:
            messaging.send(messaging.Message(
                notification=messaging.Notification(title=alert["title"], body=alert["body"]),
                token=token,
            ))
            doc.reference.set({
                "lastAlertKey": alert["key"],
                "lastAlertSentAt": admin_firestore.SERVER_TIMESTAMP,
            }, merge=True)
            alerted += 1
        except Exception as e:
            errors.append(f"{doc.id}: send failed ({str(e)})")

    return {"checked": checked, "alerted": alerted, "errors": errors}
