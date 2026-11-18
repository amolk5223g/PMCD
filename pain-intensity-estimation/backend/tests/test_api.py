"""Automated end-to-end tests for FastAPI backend endpoints using actual PMPDB data & checkpoints."""
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_and_model_status():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["models_ready"] is True

    ms = client.get("/api/models/status")
    assert ms.status_code == 200
    ms_body = ms.json()
    assert ms_body["loaded_checkpoints"]["mm_gated_dl"] is True
    assert ms_body["loaded_checkpoints"]["xgboost_baseline"] is True
    assert "ECG" in ms_body["absent_modalities"]
    assert "EEG" in ms_body["absent_modalities"]


def test_dataset_endpoints():
    ds = client.get("/api/dataset/summary")
    assert ds.status_code == 200
    assert ds.json()["num_participants"] == 49
    assert ds.json()["num_main_recordings"] == 90

    parts = client.get("/api/dataset/participants")
    assert parts.status_code == 200
    assert parts.json()["num_participants"] == 49

    rec = client.get("/api/dataset/recordings/P01_1?downsample_hz=5")
    assert rec.status_code == 200
    rec_body = rec.json()
    assert len(rec_body["signals"]) > 100
    assert len(rec_body["observed_pain_events"]) == 12


def test_prediction_and_missing_modality_handling():
    # Full 6 modalities
    r_full = client.post(
        "/api/predict/recording",
        json={
            "recording_id": "P01_1",
            "timestamp_sec": 56.46,
            "model_choice": "hybrid",
            "disabled_modalities": [],
            "include_explainability": True,
        },
    )
    assert r_full.status_code == 200
    full_body = r_full.json()
    assert 0.0 <= full_body["predicted_pain_0_100"] <= 100.0
    assert len(full_body["missing_modalities"]) == 0
    assert full_body["explainability"]["shap"] is not None
    assert full_body["explainability"]["integrated_gradients"] is not None

    # Simulated sensor dropout (EDA + BVP missing -> zero fabrication, widened uncertainty)
    r_miss = client.post(
        "/api/predict/recording",
        json={
            "recording_id": "P01_1",
            "timestamp_sec": 56.46,
            "model_choice": "mm_gated",
            "disabled_modalities": ["EDA", "BVP"],
            "include_explainability": True,
        },
    )
    assert r_miss.status_code == 200
    miss_body = r_miss.json()
    assert set(miss_body["missing_modalities"]) == {"EDA", "BVP"}
    assert miss_body["missing_modality_warning"] is not None
    assert miss_body["sensor_status"]["EDA"]["available"] is False
    assert miss_body["sensor_status"]["BVP"]["available"] is False


def test_timeline_evaluation_and_global_xai():
    tl = client.post(
        "/api/predict/timeline",
        json={"recording_id": "P01_1", "step_sec": 10.0, "model_choice": "hybrid"},
    )
    assert tl.status_code == 200
    assert len(tl.json()["timeline"]) >= 10

    ev = client.get("/api/evaluation/summary")
    assert ev.status_code == 200
    assert len(ev.json()["model_comparison_table"]) >= 9

    gx = client.get("/api/explain/global")
    assert gx.status_code == 200
    assert len(gx.json()["shap_top_25_features"]) == 25


if __name__ == "__main__":
    test_health_and_model_status()
    test_dataset_endpoints()
    test_prediction_and_missing_modality_handling()
    test_timeline_evaluation_and_global_xai()
    print("ALL BACKEND API INTEGRATION TESTS PASSED!")

