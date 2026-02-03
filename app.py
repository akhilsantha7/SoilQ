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

class AdviceItem(BaseModel):
    heading: str
    text: str

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
    advice: List[AdviceItem]  # <-- now a list of advice items

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
    try:
        if req.advice_type == "warmup":
            return AdviceResponse(advice=[AdviceItem(heading="Warmup", text="Warmup done ✅")])
        elif req.advice_type == "irrigation":
            return await irrigation_advice(req)
        elif req.advice_type == "disease":
            return await disease_advice(req)
        else:
            raise HTTPException(status_code=400, detail="Invalid advice_type")
    except Exception as e:
        # Always return valid JSON for Swift
        return JSONResponse(
            status_code=200,
            content={"advice": [{"heading": "Error", "text": f"Server error: {str(e)}"}]}
        )

# -----------------------------
# Irrigation Advice
# -----------------------------
async def irrigation_advice(req: AdviceRequest):
    import json

    forecast_text = "\n".join(
        f"- {d.date}: {d.temp:.1f}°C, {d.humidity:.0f}% humidity, {d.wind:.1f} m/s wind, {d.condition}"
        for d in req.forecast
    ) or "No forecast available"

    lang_map = {"english": "English", "hindi": "Hindi", "telugu": "Telugu"}
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
- Provide 5 advice points with headings:
  1. Current Soil Status
  2. Timing Based on Forecast
  3. Temperature Consideration
  4. Wind & Humidity Adjustment
  5. Preventive Moisture Management
- Return **JSON array**: [{"heading": "...", "text": "..."}]
- Each "text": 1–2 sentences
- Do NOT include extra text outside JSON
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        advice_text = response.choices[0].message.content.strip()

        try:
            advice_list = json.loads(advice_text)
            if not isinstance(advice_list, list):
                raise ValueError()
            advice_items = [AdviceItem(**item) for item in advice_list]
        except Exception:
            # fallback
            advice_items = [
                AdviceItem(heading=h, text="No advice available.") for h in [
                    "Current Soil Status",
                    "Timing Based on Forecast",
                    "Temperature Consideration",
                    "Wind & Humidity Adjustment",
                    "Preventive Moisture Management"
                ]
            ]

        return AdviceResponse(advice=advice_items)

    except Exception as e:
        return AdviceResponse(advice=[AdviceItem(heading="Error", text=str(e))])

# -----------------------------
# Disease Advice
# -----------------------------
async def disease_advice(req: AdviceRequest):
    import json

    forecast_text = "\n".join(
        f"- {d.date}: {d.temp:.1f}°C, {d.humidity:.0f}% humidity, {d.wind:.1f} m/s wind, {d.condition}"
        for d in req.forecast
    ) or "No forecast available"

    lang_map = {"english": "English", "hindi": "Hindi", "telugu": "Telugu"}
    lang = lang_map.get(req.language.lower(), "English")

    headings = [
        "Disease Overview",
        "Immediate Actions",
        "Control Options",
        "Weather Considerations",
        "Prevention Tips"
    ]

    prompt = f"""
You are a professional plant pathologist.
Respond in {lang}.
### Crop & Disease Info
- Crop: {req.crop_name or "Unknown"}
- Detected Disease: {req.disease_name or "Unknown"}
- Confidence: {int((req.disease_confidence or 0)*100)}%
### 7-Day Weather Forecast
{forecast_text}
### Instructions
- Provide 5 advice points with headings: {', '.join(headings)}
- Return **JSON array**: [{"heading": "...", "text": "..."}]
- Each "text": 1–2 sentences
- Do NOT include extra text outside JSON
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        advice_text = response.choices[0].message.content.strip()

        try:
            advice_list = json.loads(advice_text)
            if not isinstance(advice_list, list):
                raise ValueError()
            advice_items = [AdviceItem(**item) for item in advice_list]
        except Exception:
            advice_items = [AdviceItem(heading=h, text="No advice available.") for h in headings]

        return AdviceResponse(advice=advice_items)

    except Exception as e:
        return AdviceResponse(advice=[AdviceItem(heading="Error", text=str(e))])
