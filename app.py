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


class AdviceResponse(BaseModel):
    advice: str


# -----------------------------
# Root (Render health check)
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

    if req.advice_type == "disease":
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

### Current Field Conditions
- Crop: {req.crop_name}
- Soil Moisture: {req.soil_moisture}%
- Soil Temperature: {req.soil_temp}°C
- Soil pH: {req.soil_ph}
- Irrigation Needed: {"YES" if req.irrigation_needed == 1 else "NO"}
- Time to Irrigation: {req.time_to_irrigation} hours

### 7-Day Weather Forecast (very important)
{forecast_text}

### Rules (must follow strictly)
- If rain probability is HIGH in the next 24–48 hours, advise DELAYING irrigation.
- If no rain is expected and soil moisture is LOW, advise IMMEDIATE irrigation.
- Always mention specific days or times (example: "tomorrow morning", "within 6 hours").
- Base irrigation quantity and timing on temperature, humidity, and wind.
- DO NOT give generic advice.
- DO NOT say “monitor weather” or “keep an eye”.
- DO NOT mention AI, ML, or predictions.

### Output format (plain text, short paragraphs):
1. Clear decision: Irrigate NOW / Delay irrigation
2. Exact timing (hours or day)
3. Reason using forecast (rain, temperature, humidity)
4. Water-saving tips based on forecast
5. Risk warning if irrigation is skipped or delayed
"""


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    return AdviceResponse(
        advice=response.choices[0].message.content.strip()
    )


# -----------------------------
# Disease Advice
# -----------------------------
# -----------------------------
# Disease Advice
# -----------------------------
async def disease_advice(req: AdviceRequest):

    # Format 7-day forecast
    forecast_text = "\n".join(
        f"- {d.date}: {d.temp}°C, {d.humidity}% humidity, "
        f"{d.wind} m/s wind, {d.condition}"
        for d in req.forecast
    )

    # Focused prompt: only crop, disease, confidence, and weather
    prompt = f"""
You are an expert plant pathologist.

### Crop & Disease Info
- Crop: {req.crop_name}
- Detected Disease: {req.disease_name}
- Confidence: {int((req.disease_confidence or 0) * 100)}%

### 7-Day Weather Forecast
{forecast_text}

### Instructions
Explain for farmers:
1. What this disease is
2. Immediate treatment steps
3. Organic + chemical control options
4. How weather may affect spread
5. Prevention for next season

Use simple, clear language.
Respond ONLY in the requested format. Do NOT include AI, ML, or soil info.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=350
    )

    return AdviceResponse(
        advice=response.choices[0].message.content.strip()
    )
