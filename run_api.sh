#!/bin/bash

echo "============================================================"
echo "STARTING FRAUD DETECTION API"
echo "============================================================"
echo ""
echo "Starting server at http://localhost:8000"
echo ""
echo "Documentation:"
echo "  - Swagger UI: http://localhost:8000/docs"
echo "  - ReDoc: http://localhost:8000/redoc"
echo ""
echo "Endpoints:"
echo "  - Health: http://localhost:8000/health"
echo "  - Predict: http://localhost:8000/predict"
echo "  - Batch: http://localhost:8000/predict/batch"
echo ""
echo "============================================================"
echo "Press Ctrl+C to stop the server"
echo "============================================================"
echo ""

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

