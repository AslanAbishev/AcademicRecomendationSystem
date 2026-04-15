from fastapi import FastAPI, HTTPException

from app.config import settings
from app.services.recommender import ECRRecommenderService


app = FastAPI(title=settings.app_name, version="0.1.0")
service = ECRRecommenderService()


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.get("/researchers")
def list_researchers():
    return service.list_researchers()


@app.get("/researchers/{researcher_id}/dashboard")
def researcher_dashboard(researcher_id: int, top_k: int = 6):
    try:
        return service.get_dashboard(researcher_id, top_k=top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/researchers/{researcher_id}/collaborators")
def collaborator_recommendations(researcher_id: int, top_k: int = 4):
    try:
        return service.recommend_collaborators(researcher_id, top_k=top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
