from app.services.recommender import ECRRecommenderService


def test_collaborator_scores_exclude_self():
    service = ECRRecommenderService()
    recommendations = service.recommend_collaborators(1, top_k=3)

    ids = [item.researcher_id for item in recommendations]
    assert 1 not in ids
    assert len(recommendations) == 3


def test_ecr_dashboard_contains_fairness_signal():
    service = ECRRecommenderService()
    dashboard = service.get_dashboard(3, top_k=4)

    assert dashboard.analytics.ecr_status is True
    assert any(item.explanation.ecr_boost > 0 for item in dashboard.opportunities)


def test_cold_start_profile_still_gets_recommendations():
    service = ECRRecommenderService()
    dashboard = service.get_dashboard(3, top_k=4)

    assert dashboard.analytics.cold_start_risk > 0.5
    assert len(dashboard.opportunities) == 4
