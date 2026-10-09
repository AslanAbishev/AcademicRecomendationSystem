from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests

from app.config import settings
from app.schemas import (
    AccountRegistrationRequest,
    AuthSessionResponse,
    AuthenticatedCabinetResponse,
    AuthenticatedUser,
    CollaboratorScope,
    EmbeddingModelName,
    ExperimentPlanResponse,
    PromotionActionUpdate,
    PromotionProgressResponse,
    PromotionSnapshot,
    RegisteredResearcherProfile,
    ResearcherRegistrationRequest,
    UserLoginRequest,
)
from app.services.account_service import AccountService
from app.services.experiment_service import ExperimentService
from app.services.progress_service import PromotionProgressService
from app.services.registration_service import RegistrationService
from app.services.realtime_recommender import ECRRecommenderService
from app.services.research_database import ResearchDatabase
from app.services.scientific_embeddings import ScientificEmbeddingModel


app = FastAPI(title=settings.app_name, version="0.1.0")
service = ECRRecommenderService()
registration_service = RegistrationService(settings.registration_store_path, client=service.client)
account_service = AccountService(settings.user_store_path)
progress_service = PromotionProgressService(
    settings.promotion_snapshot_store_path,
    settings.promotion_action_store_path,
)
experiment_service = ExperimentService()
research_db = ResearchDatabase(settings.database_url)
frontend_dir = Path(__file__).resolve().parent / "frontend"

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


def get_bearer_token(authorization: str | None = Header(default=None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    return token


def get_current_user(token: str = Depends(get_bearer_token)) -> AuthenticatedUser:
    try:
        return account_service.get_user_by_token(token)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.post("/auth/register", response_model=AuthSessionResponse)
def register_account(payload: AccountRegistrationRequest):
    try:
        return account_service.register(payload, registration_service)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/auth/login", response_model=AuthSessionResponse)
def login(payload: UserLoginRequest):
    try:
        return account_service.login(payload, registration_service)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Linked researcher profile was not found") from exc


@app.get("/auth/me", response_model=AuthenticatedUser)
def get_me(current_user: AuthenticatedUser = Depends(get_current_user)):
    return current_user


@app.get("/auth/me/cabinet", response_model=AuthenticatedCabinetResponse)
def get_my_cabinet(
    current_user: AuthenticatedUser = Depends(get_current_user),
    top_k: int = 6,
    collaborator_scope: CollaboratorScope = "global",
    embedding_model: EmbeddingModelName = "hashing",
):
    try:
        profile = registration_service.get_profile(current_user.profile_id)
        return AuthenticatedCabinetResponse(
            user=current_user,
            cabinet=service.get_user_cabinet(
                profile,
                top_k=top_k,
                collaborator_scope=collaborator_scope,
                embedding_model=embedding_model,
            ),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="OpenAlex cabinet data is temporarily unavailable") from exc


@app.post("/auth/me/snapshots", response_model=PromotionSnapshot)
def create_my_promotion_snapshot(
    current_user: AuthenticatedUser = Depends(get_current_user),
    top_k: int = 6,
    collaborator_scope: CollaboratorScope = "global",
    embedding_model: EmbeddingModelName = "hashing",
):
    try:
        profile = registration_service.get_profile(current_user.profile_id)
        cabinet = service.get_user_cabinet(
            profile,
            top_k=top_k,
            collaborator_scope=collaborator_scope,
            embedding_model=embedding_model,
        )
        return progress_service.create_snapshot(profile, cabinet)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="OpenAlex snapshot data is temporarily unavailable") from exc


@app.get("/auth/me/progress", response_model=PromotionProgressResponse)
def get_my_promotion_progress(current_user: AuthenticatedUser = Depends(get_current_user)):
    return progress_service.get_progress(current_user.profile_id)


@app.post("/auth/me/actions")
def record_my_promotion_action(
    payload: PromotionActionUpdate,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, str]:
    return progress_service.record_action(current_user.profile_id, payload)


@app.post("/auth/logout")
def logout(token: str = Depends(get_bearer_token)) -> dict[str, str]:
    account_service.revoke_token(token)
    return {"status": "logged_out"}


@app.get("/experiments/ml-plan", response_model=ExperimentPlanResponse)
def get_experiment_ml_plan():
    return experiment_service.get_ml_plan()


@app.get("/experiments/dataset-stats")
def get_research_dataset_stats():
    try:
        return {**research_db.stats(), "embeddings_by_model": research_db.embedding_counts_by_model()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Research database is not available") from exc


@app.get("/experiments/vector-search")
def vector_search(q: str, limit: int = 10, model: EmbeddingModelName = "hashing"):
    try:
        query_embedding = ScientificEmbeddingModel(model).encode([q])[0].tolist()
        return {
            "query": q,
            "model": model,
            "results": research_db.search_similar_works(query_embedding, model_name=model, limit=limit),
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Vector search is not available") from exc


@app.get("/researchers/search")
def search_researchers(q: str, limit: int = 8):
    try:
        return service.search_researchers(q, limit=limit)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="OpenAlex search is temporarily unavailable") from exc


@app.get("/researchers/{researcher_id}/dashboard")
def researcher_dashboard(
    researcher_id: str,
    top_k: int = 6,
    collaborator_scope: CollaboratorScope = "global",
    embedding_model: EmbeddingModelName = "hashing",
):
    try:
        return service.get_dashboard(
            researcher_id,
            top_k=top_k,
            collaborator_scope=collaborator_scope,
            embedding_model=embedding_model,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="OpenAlex dashboard data is temporarily unavailable") from exc


@app.get("/researchers/{researcher_id}/collaborators")
def collaborator_recommendations(
    researcher_id: str,
    top_k: int = 4,
    collaborator_scope: CollaboratorScope = "global",
    embedding_model: EmbeddingModelName = "hashing",
):
    try:
        return service.get_dashboard(
            researcher_id,
            top_k=top_k,
            collaborator_scope=collaborator_scope,
            embedding_model=embedding_model,
        ).collaborators
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="OpenAlex collaborator data is temporarily unavailable") from exc


@app.post("/profiles/register", response_model=RegisteredResearcherProfile)
def register_profile(payload: ResearcherRegistrationRequest):
    return registration_service.register(payload)


@app.get("/profiles/registered", response_model=list[RegisteredResearcherProfile])
def list_registered_profiles():
    return registration_service.list_profiles()


@app.get("/profiles/registered/{profile_id}", response_model=RegisteredResearcherProfile)
def get_registered_profile(profile_id: str):
    try:
        return registration_service.get_profile(profile_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/profiles/registered/{profile_id}/dashboard")
def get_registered_profile_dashboard(
    profile_id: str,
    top_k: int = 6,
    collaborator_scope: CollaboratorScope = "global",
    embedding_model: EmbeddingModelName = "hashing",
):
    try:
        profile = registration_service.get_profile(profile_id)
        return service.get_registered_profile_dashboard(
            profile,
            top_k=top_k,
            collaborator_scope=collaborator_scope,
            embedding_model=embedding_model,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="OpenAlex dashboard data is temporarily unavailable") from exc


@app.get("/profiles/registered/{profile_id}/cabinet")
def get_registered_profile_cabinet(
    profile_id: str,
    top_k: int = 6,
    collaborator_scope: CollaboratorScope = "global",
    embedding_model: EmbeddingModelName = "hashing",
):
    try:
        profile = registration_service.get_profile(profile_id)
        return service.get_user_cabinet(
            profile,
            top_k=top_k,
            collaborator_scope=collaborator_scope,
            embedding_model=embedding_model,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="OpenAlex cabinet data is temporarily unavailable") from exc
