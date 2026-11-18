"""FastAPI application entrypoint for Explainable Multimodal Continuous Pain-Intensity Estimation."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.config import FIGURES_DIR

app = FastAPI(
    title="Explainable Multimodal Continuous Pain-Intensity Estimation API",
    description=(
        "Research-grade biomedical signal processing, multimodal deep learning & ML inference, "
        "MC Dropout uncertainty quantification, and SHAP/Integrated Gradients explainability "
        "trained on the local PMPDB (PainMonit Physiotherapy Database). "
        "For academic research only — not for clinical diagnosis or treatment."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if FIGURES_DIR.exists():
    app.mount("/reports/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


@app.get("/")
def root():
    return {
        "name": "Explainable Multimodal Continuous Pain-Intensity Estimation API",
        "docs_url": "/docs",
        "health_url": "/api/health",
    }

