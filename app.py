# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import openai
# import os
# from openai import OpenAI


# app = FastAPI()

# # Set API key via environment variable
# # openai.api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# # Request model
# class PredictionRequest(BaseModel):
#     irrigation_needed: float
#     time_to_irrigation: float
#     fertilizer_type: str
#     weather_temp: float
#     weather_humidity: float
#     weather_wind: float
#     weather_condition: str

# # Response model
# class GenAIResponse(BaseModel):
#     advice: str

# @app.post("/genai", response_model=GenAIResponse)
# async def get_genai_advice(pred: PredictionRequest):
#     prompt = f"""
#     Farmer has these conditions and predictions:
#     - Irrigation Needed: {'Yes' if pred.irrigation_needed == 1 else 'No'}
#     - Time to Irrigation: {pred.time_to_irrigation} hrs
#     - Fertilizer Type: {pred.fertilizer_type}

#     Current weather conditions:
#     - Temperature: {pred.weather_temp} °C
#     - Humidity: {pred.weather_humidity} %
#     - Wind Speed: {pred.weather_wind} m/s
#     - Condition: {pred.weather_condition}

#     Provide actionable advice for the farmer considering both sensor predictions and current weather.
#     """

#     try:
#         response = openai.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=200
#         )
#         advice = response.choices[0].message.content.strip()
#         return {"advice": advice}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import os
import json

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DailyForecast(BaseModel):
    date: str
    temp: float
    humidity: float
    wind: float
    condition: str

class PredictionRequest(BaseModel):
    irrigation_needed: float
    time_to_irrigation: float
    fertilizer_type: str
    forecast: List[DailyForecast]

class AdviceResponse(BaseModel):
    telugu: str
    hindi: str
    english: str


@app.post("/genai", response_model=AdviceResponse)
async def get_genai_advice(pred: PredictionRequest):

    forecast_text = "\n".join([
        f"- {day.date}: {day.temp}°C, {day.humidity}% humidity, "
        f"{day.wind} m/s wind, {day.condition}"
        for day in pred.forecast
    ])

    prompt = f"""
You are an expert agriculture assistant.

Use the following data:

**Predictions**:
- Irrigation Needed: {"Yes" if pred.irrigation_needed == 1 else "No"}
- Time to Irrigation: {pred.time_to_irrigation} hours
- Fertilizer Type: {pred.fertilizer_type}

**7-Day Weather Forecast**:
{forecast_text}

Respond ONLY in the following strict JSON format:

{{
  "telugu": "<advice in telugu>",
  "hindi": "<advice in hindi>",
  "english": "<advice in english>"
}}

DO NOT include anything outside the JSON.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        raw = response.choices[0].message.content.strip()

        # Parse the JSON returned by GPT safely
        data = json.loads(raw)

        return AdviceResponse(
            telugu=data.get("telugu", ""),
            hindi=data.get("hindi", ""),
            english=data.get("english", "")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
