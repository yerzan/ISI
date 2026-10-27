"""Testy lab3: silnik CF + endpointy Flask. Uruchom: py -m pytest test_lab03.py -q"""
import sys
import pytest

from app import app
from recommender import (
    ratings_matrix as matrix,
    recommend_collaborative as recommend,
    similar_users_collaborative as similar_users,
    similarity_matrix as sim,
)

client = app.test_client()


def test_matrix_shape():
    assert matrix.shape == (12, 24)
    assert matrix.index.is_unique and matrix.columns.is_unique


def test_symetria_i_diagonala():
    for u in matrix.index:
        assert sim.loc[u, u] == pytest.approx(1.0, abs=1e-9)
        for v in matrix.index:
            assert sim.loc[u, v] == pytest.approx(sim.loc[v, u], abs=1e-9)


def test_rekomendacje_maja_skale():
    for u in matrix.index:
        for r in recommend(u):
            assert 1.0 <= r["predicted"] <= 5.0


def test_nie_polecamy_obejrzanego():
    for u in ["Ania", "Gosia", "Jacek"]:
        rated = set(matrix.columns[matrix.loc[u].notna()])
        for r in recommend(u):
            assert r["title"] not in rated


def test_znany_user():
    recs = recommend("Ania")
    assert recs and len(recs) == 5
    assert all("because" in r for r in recs)


def test_nieznany_user():
    assert recommend("Zzz") is None
    assert similar_users("Zzz") is None


def test_sasiedzi_z_tego_samego_plemienia():
    # Ania, Bartek i Celina to fani sci-fi; Daria/Ewa/Filip to romans
    assert similar_users("Ania", k=1)[0]["user"] in {"Bartek", "Celina"}
    assert similar_users("Daria", k=1)[0]["user"] in {"Ewa", "Filip"}


def test_api_recommend_200():
    r = client.get("/api/recommend/collaborative?user=Ania")
    assert r.status_code == 200
    data = r.get_json()
    assert data["user"] == "Ania"
    assert len(data["recommendations"]) == 5
    assert data["recommendations"][0]["predicted"] >= data["recommendations"][-1]["predicted"]


def test_api_top_n():
    r = client.get("/api/recommend/collaborative?user=Gosia&top_n=2")
    assert r.status_code == 200
    assert len(r.get_json()["recommendations"]) == 2


def test_api_400_bez_parametru():
    assert client.get("/api/recommend/collaborative").status_code == 400


def test_api_404_nieznany():
    r = client.get("/api/recommend/collaborative?user=Zzz")
    assert r.status_code == 404
    assert "Ania" in r.get_json()["known_users"]


def test_api_users():
    assert len(matrix.index) == 12


def test_api_similar():
    similar = similar_users("Jacek")

    assert similar is not None
    assert len(similar) > 0
    assert similar[0]["user"] in {"Leon", "Kasia"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
