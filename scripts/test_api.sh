#!/usr/bin/env bash
# Test SoilQ GenAI API locally. Start server first: uvicorn app:app --reload
set -e
BASE="${1:-https://soilq.onrender.com}"

echo "=== Testing base URL: $BASE ==="

echo ""
echo "1. Health check GET /"
curl -s "$BASE/" | head -1

echo ""
echo "2. Warmup POST /genai"
curl -s -X POST "$BASE/genai" \
  -H "Content-Type: application/json" \
  -d '{"advice_type":"warmup"}' | head -1

echo ""
echo "3. Disease advice (text) POST /genai"
curl -s -X POST "$BASE/genai" \
  -H "Content-Type: application/json" \
  -d '{
    "advice_type":"disease",
    "crop_name":"Tomato",
    "disease_name":"Early blight",
    "disease_confidence":0.85,
    "language":"english",
    "forecast":[{"date":"2026-02-25","temp":28,"humidity":70,"wind":2,"condition":"Sunny"}]
  }' | head -3

echo ""
echo "4. Irrigation advice POST /genai"
curl -s -X POST "$BASE/genai" \
  -H "Content-Type: application/json" \
  -d '{
    "advice_type":"irrigation",
    "crop_name":"Tomato",
    "soil_moisture":42,
    "soil_temp":27,
    "soil_ph":6.5,
    "irrigation_needed":0,
    "language":"english",
    "forecast":[{"date":"2026-02-25","temp":29,"humidity":65,"wind":3,"condition":"Sunny"}]
  }' | head -3

echo ""
echo "5. Disease from image POST /genai/disease-from-image"
echo "   (skipped if no image; run manually with a plant image)"
if [ -n "$2" ] && [ -f "$2" ]; then
  curl -s -X POST "$BASE/genai/disease-from-image" \
    -F "image=@$2" \
    -F "crop_name=Tomato" | head -5
else
  echo "   Usage: $0 $BASE /path/to/leaf.jpg"
fi

echo ""
echo "=== Done ==="
