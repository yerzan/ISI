from flask import Flask, jsonify, request

app = Flask(__name__)

PRODUCTS = [
    {"id": 1, "nazwa": "Klawiatura mechaniczna", "kategoria": "akcesoria", "cena": 349.99, "opis": "Przełączniki hot-swap, podświetlenie RGB"},
    {"id": 2, "nazwa": "Mysz bezprzewodowa", "kategoria": "akcesoria", "cena": 129.00, "opis": "Ciche przyciski, 4000 DPI"},
    {"id": 3, "nazwa": "Monitor 27 cali", "kategoria": "monitory", "cena": 1099.00, "opis": "Matryca IPS, 144 Hz"},
    {"id": 4, "nazwa": "Słuchawki nauszne", "kategoria": "audio", "cena": 449.00, "opis": "ANC, 40 h pracy na baterii"},
    {"id": 5, "nazwa": "Hub USB-C", "kategoria": "akcesoria", "cena": 199.00, "opis": "HDMI, 3x USB-A, czytnik SD"},
]

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/products")
def products():
    return jsonify(PRODUCTS)

@app.route("/api/products/<int:id>")
def product(id):
    for product in PRODUCTS:
        if product["id"] == id:
            return jsonify(product)

    return jsonify({"error": "not found"}), 404


@app.route("/api/products", methods=["POST"])
def add_product():
    data = request.get_json()

    new_id = max(product["id"] for product in PRODUCTS) + 1

    new_product = {
        "id": new_id,
        "nazwa": data["nazwa"],
        "kategoria": data["kategoria"],
        "cena": data["cena"],
        "opis": data["opis"]
    }

    PRODUCTS.append(new_product)

    return jsonify(new_product), 201

if __name__ == "__main__":
    app.run(debug=True)