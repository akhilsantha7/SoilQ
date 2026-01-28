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
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        advice_text = response.choices[0].message.content.strip()

        # Attempt to parse JSON array
        import json
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
