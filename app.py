from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional
from openai import OpenAI
from fastapi.responses import JSONResponse
import os

# -----------------------------
# App + OpenAI Client
# -----------------------------
app = FastAPI(title="SoilQ GenAI Service")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
# -----------------------------
# Irrigation Advice (Updated for iOS)
# -----------------------------
# -----------------------------
# Irrigation Advice (Fixed)
# -----------------------------
# -----------------------------
# Irrigation Advice (like Disease Advice)
# -----------------------------
# -----------------------------
# Irrigation Advice (Full Working)
# -----------------------------
async def irrigation_advice(req: AdviceRequest):
    import json
    from fastapi.responses import JSONResponse

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

    # Headings for irrigation advice
    headings = [
        "Current Soil Status",
        "Timing Based on Forecast",
        "Temperature Consideration",
        "Wind & Humidity Adjustment",
        "Preventive Moisture Management"
    ]

    # Prompt for AI
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
- Provide 5 clear advice points for farmers.
- Use the following headings exactly:
    {', '.join(headings)}
- Each point should have a heading and explanation.
- Focus on prediction-driven, actionable advice based on current soil and forecast.
- Respond **only in JSON** as an array of objects: 
  [{"heading": "...", "text": "..."}]
- Each "text" should be 1–2 sentences.
- Do NOT include AI mentions or extra text outside JSON.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        advice_text = response.choices[0].message.content.strip()

        # Attempt to parse JSON
        advice_json = []
        try:
            advice_json = json.loads(advice_text)
            # Validate: must be list of dicts with heading+text
            if not isinstance(advice_json, list) or not all(
                isinstance(a, dict) and "heading" in a and "text" in a for a in advice_json
            ):
                raise ValueError("Invalid JSON structure")
        except Exception:
            # Fallback: generic placeholders
            advice_json = [{"heading": h, "text": "No advice available."} for h in headings]

        return JSONResponse(content={"advice": advice_json})

    except Exception as e:
        # Catch-all error
        advice_json = [{"heading": "Error", "text": f"Error generating advice: {str(e)}"}]
        return JSONResponse(content={"advice": advice_json})


# -----------------------------
# Disease Advice
# -----------------------------
# -----------------------------
# Disease Advice (Updated for iOS)
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

    # Headings for disease advice
    headings = [
        "Disease Overview",
        "Immediate Actions",
        "Control Options",
        "Weather Considerations",
        "Prevention Tips"
    ]

    # Prompt for AI
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
    {', '.join(headings)}
- Return a **JSON array of objects**: 
  [{{"heading": "...", "text": "..."}}, ...]
- Each advice text should be 1–2 sentences.
- Do NOT include AI mentions or extra text outside JSON.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        advice_text = response.choices[0].message.content.strip()

        import json
        advice_json = []
        try:
            advice_json = json.loads(advice_text)
            # Validate: must be list of dicts with heading+text
            if not isinstance(advice_json, list) or not all(
                isinstance(a, dict) and "heading" in a and "text" in a for a in advice_json
            ):
                raise ValueError("Invalid JSON structure")
        except Exception:
            # Fallback if parsing fails
            advice_json = [{"heading": h, "text": "No advice available."} for h in headings]

        return JSONResponse(content={"advice": advice_json})

    except Exception as e:
        return JSONResponse(content={
            "advice": [{"heading": "Error", "text": f"Error generating advice: {str(e)}"}]
        })
