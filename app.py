from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os, json

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------- Models -----------------
class DailyForecast(BaseModel):
    date: str
    temp: float
    humidity: float
    wind: float
    condition: str

class DiseaseAdviceRequest(BaseModel):
    advice_type: str                  # "disease" or "soil"
    plant_disease: str
    disease_confidence: float
    plant_type: str
    soil_moisture: Optional[float] = None
    soil_temp: Optional[float] = None
    soil_ph: Optional[float] = None
    forecast: List[DailyForecast]

class AdviceResponse(BaseModel):
    telugu: str
    hindi: str
    english: str

# ----------------- Endpoint -----------------
@app.post("/genai", response_model=AdviceResponse)
async def get_disease_advice(req: DiseaseAdviceRequest):
    if req.advice_type != "disease":
        raise HTTPException(status_code=400, detail="Only 'disease' advice supported here")

    # Format 7-day forecast
    forecast_text = "\n".join([
        f"- {day.date}: {day.temp}°C, {day.humidity}% humidity, "
        f"{day.wind} m/s wind, {day.condition}" for day in req.forecast
    ])

    # Prompt for OpenAI
    prompt = f"""
You are an expert plant pathologist.

The farmer's crop has the following confirmed disease:
- Crop: {req.plant_type}
- Disease: {req.plant_disease}
- Confidence: {int(req.disease_confidence * 100)}%

Soil conditions:
- Moisture: {req.soil_moisture or 'N/A'}%
- Temperature: {req.soil_temp or 'N/A'}°C
- pH: {req.soil_ph or 'N/A'}

7-day weather forecast:
{forecast_text}

Instructions:
Explain clearly for farmers:
1. What this disease is
2. Immediate treatment steps
3. Organic and chemical control options
4. How weather may affect disease spread
5. Prevention for next season

Respond ONLY in strict JSON format:
{{
  "telugu": "<advice in Telugu>",
  "hindi": "<advice in Hindi>",
  "english": "<advice in English>"
}}
Do NOT include anything outside the JSON.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )

        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)

        return AdviceResponse(
            telugu=data.get("telugu", ""),
            hindi=data.get("hindi", ""),
            english=data.get("english", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
