from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional
from openai import OpenAI
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
    advice_type: Literal["irrigation", "disease"]

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
    forecast: List[DailyForecast]

    language: Optional[str] = "english"


class AdviceResponse(BaseModel):
    advice: str


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health():
    return {"status": "SoilQ GenAI is running 🌱"}


# -----------------------------
# Main API
# -----------------------------
@app.post("/genai", response_model=AdviceResponse)
async def generate_advice(req: AdviceRequest):
    if req.advice_type == "irrigation":
        return await irrigation_advice(req)
    elif req.advice_type == "disease":
        return await disease_advice(req)
    raise HTTPException(status_code=400, detail="Invalid advice_type")


# -----------------------------
# Irrigation Advice
# -----------------------------
async def irrigation_advice(req: AdviceRequest):
    forecast_text = "\n".join(
        f"- {d.date}: {d.temp}°C, {d.humidity}% humidity, "
        f"{d.wind} m/s wind, {d.condition}"
        for d in req.forecast
    )

    prompt = f"""
You are an expert irrigation advisor for farmers.
Respond in {req.language.capitalize() if req.language else "English"} in a professional, clear style suitable for farmers.

### Current Field Conditions
- Crop: {req.crop_name or "Unknown"}
- Soil Moisture: {req.soil_moisture or 0}%
- Soil Temperature: {req.soil_temp or 0}°C
- Soil pH: {req.soil_ph or 0}
- Irrigation Needed: {"YES" if (req.irrigation_needed or 0) == 1 else "NO"}
- Time to Irrigation: {req.time_to_irrigation or 0} hours

### 7-Day Weather Forecast
{forecast_text or "No forecast available"}

### Rules
- Advise exact timing (e.g., "within 6 hours", "tomorrow morning").
- If rain probability is HIGH in next 48 hours, advise delaying irrigation.
- If soil moisture is LOW and no rain, advise immediate irrigation.
- Provide clear reasoning using weather data.
- Include water-saving tips and risk warnings if delayed.
- DO NOT give generic advice or mention AI, ML, or predictions.

### Output (plain text, short paragraphs):
- Full sentence explanation for decision.
- Timing recommendation.
- Reason based on forecast.
- Water-saving advice.
- Risk warning if skipped.
"""

    response = client.chat.completions.create(
        model="gpt-5-mini",  # ✅ switched to GPT-5 Mini
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )

    return AdviceResponse(
        advice=response.choices[0].message.content.strip()
    )


# -----------------------------
# Disease Advice
# -----------------------------
async def disease_advice(req: AdviceRequest):
    forecast_text = "\n".join(
        f"- {d.date}: {d.temp}°C, {d.humidity}% humidity, "
        f"{d.wind} m/s wind, {d.condition}"
        for d in req.forecast
    )

    prompt = f"""
You are an expert plant pathologist.

### Crop & Disease Info
- Crop: {req.crop_name}
- Detected Disease: {req.disease_name}
- Confidence: {int((req.disease_confidence or 0) * 100)}%

### 7-Day Weather Forecast
{forecast_text}

### Instructions
Explain in clear, simple, professional language suitable for farmers:
1. What this disease is
2. Immediate treatment steps
3. Organic and chemical control options
4. How weather affects spread
5. Prevention for next season

Respond ONLY in requested format. Do NOT include AI, ML, or soil info.
"""

    response = client.chat.completions.create(
        model="gpt-5-mini",  # ✅ switched to GPT-5 Mini
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )

    return AdviceResponse(
        advice=response.choices[0].message.content.strip()
    )
