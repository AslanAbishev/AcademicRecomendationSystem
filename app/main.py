from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests

from app.config import settings
from app.services.realtime_recommender import ECRRecommenderService


app = FastAPI(title=settings.app_name, version="0.1.0")
service = ECRRecommenderService()
frontend_dir = Path(__file__).resolve().parent / "frontend"

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.get("/researchers/search")
def search_researchers(q: str, limit: int = 8):
    try:
        return service.search_researchers(q, limit=limit)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="OpenAlex search is temporarily unavailable") from exc


@app.get("/researchers/{researcher_id}/dashboard")
def researcher_dashboard(researcher_id: str, top_k: int = 6):
    try:
        return service.get_dashboard(researcher_id, top_k=top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="OpenAlex dashboard data is temporarily unavailable") from exc


@app.get("/researchers/{researcher_id}/collaborators")
def collaborator_recommendations(researcher_id: str, top_k: int = 4):
    try:
        return service.get_dashboard(researcher_id, top_k=top_k).collaborators
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="OpenAlex collaborator data is temporarily unavailable") from exc
