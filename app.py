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
            model="gpt-4o-mini",
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
    forecast_text = "\n".join(
        f"- {d.date}: {d.temp:.1f}°C, {d.humidity:.0f}% humidity, "
        f"{d.wind:.1f} m/s wind, {d.condition}"
        for d in req.forecast
    ) or "No forecast available"

    lang_map = {
        "english": "English",
        "hindi": "Hindi",
        "telugu": "Telugu"
    }
    lang = lang_map.get(req.language.lower(), "English")

    prompt = f"""
You are a professional plant pathologist.

### Crop & Disease Info
- Crop: {req.crop_name or "Unknown"}
- Detected Disease: {req.disease_name or "Unknown"}
- Confidence: {int((req.disease_confidence or 0) * 100)}%

### 7-Day Weather Forecast
{forecast_text}

### Instructions
- Provide **5 concise advice points** for farmers, each with its own heading.
- Headings and points should be like:
    1. **Disease Overview**: Short explanation of the disease.
    2. **Immediate Actions**: What to do right now.
    3. **Control Options**: Organic & chemical methods.
    4. **Weather Considerations**: How forecast affects disease.
    5. **Prevention Tips**: Steps to avoid next season.
- Use complete sentences, 1–2 per point.
- Each heading should be followed by its advice on a new line.
- Return only the headings and advice points, each on its own line.
- Do not include AI, predictions, or numbered lists.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )

        advice_text = response.choices[0].message.content.strip()
        return JSONResponse(content={"advice": advice_text})

    except Exception as e:
        return JSONResponse(content={"advice": f"Error generating advice: {str(e)}"})
