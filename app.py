from math import isfinite

from flask import Flask, jsonify, request
from sklearn import neighbors

from recommender import df, recommend, recommend_collaborative, ratings_matrix, similar_users_collaborative, user_means

app = Flask(__name__)


# Laboratorium 1: przykładowe produkty
PRODUCTS = [
    {
        "id": 1,
        "nazwa": "Klawiatura mechaniczna",
        "kategoria": "akcesoria",
        "cena": 349.99,
        "opis": "Przelaczniki mechaniczne, podswietlenie RGB",
    },
    {
        "id": 2,
        "nazwa": "Mysz bezprzewodowa",
        "kategoria": "akcesoria",
        "cena": 129.00,
        "opis": "Ciche przyciski, 4000 DPI",
    },
    {
        "id": 3,
        "nazwa": "Monitor 27 cali",
        "kategoria": "monitory",
        "cena": 1099.00,
        "opis": "Matryca IPS, 144 Hz",
    },
    {
        "id": 4,
        "nazwa": "Sluchawki nauszne",
        "kategoria": "audio",
        "cena": 449.00,
        "opis": "Redukcja szumow, 40 godzin pracy",
    },
    {
        "id": 5,
        "nazwa": "Hub USB-C",
        "kategoria": "akcesoria",
        "cena": 199.00,
        "opis": "HDMI, USB-A, czytnik kart SD",
    },
]


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/products", methods=["GET"])
def api_products():
    products = PRODUCTS

    if "max_price" in request.args:
        try:
            max_price = float(request.args["max_price"])
        except (TypeError, ValueError):
            return jsonify({
                "error": "max_price musi byc poprawna liczba"
            }), 400

        if not isfinite(max_price) or max_price < 0:
            return jsonify({
                "error": "max_price musi byc liczba >= 0"
            }), 400

        products = [
            product for product in PRODUCTS
            if product["cena"] <= max_price
        ]

    return jsonify(products)


@app.route("/api/products/<int:product_id>", methods=["GET"])
def api_product_by_id(product_id):
    product = next(
        (item for item in PRODUCTS if item["id"] == product_id),
        None,
    )

    if product is None:
        return jsonify({"error": "not found"}), 404

    return jsonify(product)


@app.route("/api/products", methods=["POST"])
def api_create_product():
    if not request.is_json:
        return jsonify({
            "error": "Content-Type musi byc application/json"
        }), 400

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "JSON musi byc obiektem"}), 400

    for key in ("nazwa", "kategoria", "opis"):
        value = payload.get(key)

        if not isinstance(value, str) or not value.strip():
            return jsonify({
                "error": f"{key} musi byc niepustym tekstem"
            }), 400

    if "id" in payload:
        return jsonify({
            "error": "id jest nadawane przez serwer"
        }), 400

    price = payload.get("cena")

    if (
        isinstance(price, bool)
        or not isinstance(price, (int, float))
        or not isfinite(float(price))
        or price < 0
    ):
        return jsonify({
            "error": "cena musi byc liczba >= 0"
        }), 400

    product = {
        "id": max(
            (item["id"] for item in PRODUCTS),
            default=0
        ) + 1,
        "nazwa": payload["nazwa"].strip(),
        "kategoria": payload["kategoria"].strip(),
        "cena": float(price),
        "opis": payload["opis"].strip(),
    }

    PRODUCTS.append(product)

    return jsonify(product), 201


# Laboratorium 2: rekomendacje filmow
@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    title = request.args.get("title", "").strip()

    if not title:
        return jsonify({
            "error": "Podaj parametr ?title=..."
        }), 400

    raw_top_n = request.args.get("top_n")

    if raw_top_n is None:
        top_n = 5
    else:
        try:
            top_n = int(raw_top_n)
        except ValueError:
            return jsonify({
                "error": "top_n musi byc liczba calkowita od 1 do 10"
            }), 400

    if not 1 <= top_n <= 10:
        return jsonify({
            "error": "top_n musi byc z zakresu 1-10"
        }), 400

    genre = request.args.get("genre", "").strip() or None

    recommendations = recommend(
        title,
        top_n=top_n,
        genre=genre,
    )

    if recommendations is None:
        return jsonify({
            "error": "Nie znam takiego filmu",
            "title": title,
        }), 404

    return jsonify({
        "title": title,
        "count": len(recommendations),
        "recommendations": recommendations,
    })


@app.route("/api/movies", methods=["GET"])
def api_movies():
    movies = df[
        ["title", "genres", "overview", "rating"]
    ].to_dict(orient="records")

    return jsonify({
        "count": len(movies),
        "movies": movies,
    })

#============LAB3=========
@app.route("/api/recommend/collaborative", methods=["GET"])
def collaborative_recommendations():
    user = request.args.get("user")

    if not user:
        return jsonify({
            "error": "Missing required parameter: user"
        }), 400

    try:
        top_n = int(request.args.get("top_n", 5))
        k = int(request.args.get("k", 3))
    except ValueError:
        return jsonify({
            "error": "top_n and k must be integers"
        }), 400

    if top_n < 1:
        return jsonify({
            "error": "top_n must be greater than 0"
        }), 400

    if k < 1:
        return jsonify({
            "error": "k must be greater than 0"
        }), 400

    if user not in ratings_matrix.index:
        return jsonify({
            "error": "Unknown user",
            "known_users": list(ratings_matrix.index)
        }), 404

    recommendations = recommend_collaborative(
        user,
        top_n=top_n,
        k=k
    )

    return jsonify({
        "user": user,
        "mean_rating": round(float(user_means.loc[user]), 3),
        "count": len(recommendations),
        "recommendations": recommendations
    })

@app.route("/api/users", methods=["GET"])
def api_users():
    users = sorted(ratings_matrix.index.tolist())
    return jsonify({"count": len(users), "users": users})


@app.route("/api/similar", methods=["GET"])
def api_similar_users():
    user = request.args.get("user", "").strip()
    if not user:
        return jsonify({"error": "Podaj parametr ?user=..."}), 400

    try:
        k = int(request.args.get("k", 3))
    except ValueError:
        return jsonify({"error": "k musi byc liczba calkowita"}), 400

    if not 1 <= k <= len(ratings_matrix.index) - 1:
        return jsonify({"error": f"k musi byc z zakresu 1-{len(ratings_matrix.index) - 1}"}), 400

    neighbors = similar_users_collaborative(user, k=k)
    if neighbors is None:
        return jsonify({
            "error": "Nie znam takiego uzytkownika",
            "known_users": sorted(ratings_matrix.index.tolist())
        }), 404

    return jsonify({"user": user, "neighbors": neighbors})

if __name__ == "__main__":
    app.run(debug=True)
