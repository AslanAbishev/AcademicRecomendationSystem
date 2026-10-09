from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.schemas import (
    PromotionActionUpdate,
    PromotionProgressResponse,
    PromotionSnapshot,
    RegisteredResearcherProfile,
    UserCabinetResponse,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class PromotionProgressService:
    def __init__(self, snapshot_path: Path, action_path: Path) -> None:
        self.snapshot_path = snapshot_path
        self.action_path = action_path
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.action_path.parent.mkdir(parents=True, exist_ok=True)

    def create_snapshot(
        self,
        profile: RegisteredResearcherProfile,
        cabinet: UserCabinetResponse,
    ) -> PromotionSnapshot:
        dashboard = cabinet.personalized_dashboard
        tracker = cabinet.promotion_tracker
        derived = profile.derived_metrics
        snapshot = PromotionSnapshot(
            snapshot_id=str(uuid4()),
            profile_id=profile.profile_id,
            created_at=_utc_now(),
            promotion_score=tracker.promotion_score,
            visibility_score=profile.visibility_score,
            works_count=int(derived.get("works_count", dashboard.analytics.publication_count)),
            citation_count=int(derived.get("citation_count", dashboard.researcher.cited_by_count)),
            h_index=int(derived.get("h_index", dashboard.researcher.h_index)),
            collaborator_count=len(dashboard.collaborators),
            completed_actions=self._completed_action_count(profile.profile_id),
        )
        snapshots = self._read_snapshots()
        snapshots.append(snapshot.model_dump(mode="json"))
        self._write_json(self.snapshot_path, snapshots)
        return snapshot

    def get_progress(self, profile_id: str) -> PromotionProgressResponse:
        snapshots = [
            PromotionSnapshot.model_validate(item)
            for item in self._read_snapshots()
            if item.get("profile_id") == profile_id
        ]
        snapshots.sort(key=lambda item: item.created_at)
        baseline = snapshots[0] if snapshots else None
        latest = snapshots[-1] if snapshots else None
        deltas = self._build_deltas(baseline, latest)
        return PromotionProgressResponse(
            profile_id=profile_id,
            baseline=baseline,
            latest=latest,
            deltas=deltas,
            snapshots=snapshots,
        )

    def record_action(
        self,
        profile_id: str,
        payload: PromotionActionUpdate,
    ) -> dict[str, str]:
        actions = self._read_actions()
        actions.append(
            {
                "action_id": str(uuid4()),
                "profile_id": profile_id,
                "title": payload.title,
                "status": payload.status,
                "updated_at": _utc_now(),
            }
        )
        self._write_json(self.action_path, actions)
        return {"status": "saved", "profile_id": profile_id}

    def _completed_action_count(self, profile_id: str) -> int:
        return sum(
            1
            for item in self._read_actions()
            if item.get("profile_id") == profile_id and item.get("status") == "completed"
        )

    def _build_deltas(
        self,
        baseline: PromotionSnapshot | None,
        latest: PromotionSnapshot | None,
    ) -> dict[str, float | int]:
        if not baseline or not latest:
            return {}
        return {
            "promotion_score": round(latest.promotion_score - baseline.promotion_score, 4),
            "visibility_score": round(latest.visibility_score - baseline.visibility_score, 4),
            "works_count": latest.works_count - baseline.works_count,
            "citation_count": latest.citation_count - baseline.citation_count,
            "h_index": latest.h_index - baseline.h_index,
            "collaborator_count": latest.collaborator_count - baseline.collaborator_count,
            "completed_actions": latest.completed_actions - baseline.completed_actions,
        }

    def _read_snapshots(self) -> list[dict]:
        return self._read_json(self.snapshot_path)

    def _read_actions(self) -> list[dict]:
        return self._read_json(self.action_path)

    def _read_json(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        return json.loads(raw)

    def _write_json(self, path: Path, payload: list[dict]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
