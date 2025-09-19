from openai import OpenAI
import os
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class GenAIResponse(BaseModel):
    advice: str

class PredictionRequest(BaseModel):
    irrigation_needed: float
    time_to_irrigation: float
    fertilizer_type: str

@app.post("/genai", response_model=GenAIResponse)
async def get_genai_advice(pred: PredictionRequest):
    prompt = f"""
    Farmer has these predictions:
    - Irrigation Needed: {'Yes' if pred.irrigation_needed == 1 else 'No'}
    - Time to Irrigation: {pred.time_to_irrigation} hrs
    - Fertilizer Type: {pred.fertilizer_type}

    Provide actionable advice for the farmer.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        advice = response.choices[0].message.content.strip()
        return {"advice": advice}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
