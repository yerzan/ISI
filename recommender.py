from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path(__file__).with_name("movies.csv")
REQUIRED_COLUMNS = {"title", "genres", "overview", "rating"}


def load_data():
    df = pd.read_csv(DATA_PATH)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Brakuje kolumn w movies.csv: {sorted(missing_columns)}"
        )

    df = df.copy().reset_index(drop=True)

    for column in ("title", "genres", "overview"):
        df[column] = df[column].fillna("").astype(str)

    df["rating"] = pd.to_numeric(
        df["rating"], errors="coerce"
    ).fillna(0.0)

    return df


df = load_data()

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["overview"])

sim_matrix = cosine_similarity(tfidf_matrix)


def recommend(title, top_n=5, genre=None):
    title = title.strip().casefold()

    matches = df.index[
        df["title"].str.strip().str.casefold() == title
    ].tolist()

    if not matches:
        return None

    if not isinstance(top_n, int) or not 1 <= top_n <= 10:
        raise ValueError("top_n musi być liczbą całkowitą od 1 do 10")

    idx = matches[0]
    scores = []

    for i, score in enumerate(sim_matrix[idx]):
        if i == idx:
            continue

        if genre:
            genres = [
                item.strip().casefold()
                for item in df.iloc[i]["genres"].split("|")
            ]

            if genre.strip().casefold() not in genres:
                continue

        scores.append((i, float(score)))

    scores.sort(key=lambda x: x[1], reverse=True)

    return [
        {
            "title": df.iloc[i]["title"],
            "genres": df.iloc[i]["genres"],
            "rating": round(float(df.iloc[i]["rating"]), 1),
            "score": round(score, 3),
        }
        for i, score in scores[:top_n]
    ]

# LAB 3 - Collaborative Filtering
# Rekomendacje filmow na podstawie ocen uzytkownikow

RATINGS_PATH = Path(__file__).parent / "ratings.csv"

ratings_df = pd.read_csv(RATINGS_PATH)

# Macierz ocen:
# wiersze - uzytkownicy
# kolumny - filmy
# NaN oznacza brak oceny
ratings_matrix = ratings_df.pivot_table(
    index="user",
    columns="title",
    values="rating"
)

# Srednia ocena kazdego uzytkownika
user_means = ratings_matrix.mean(axis=1)


# Centrowanie ocen:
# od kazdej oceny odejmujemy srednia ocene uzytkownika
# brakujace wartosci uzupelniamy zerem
ratings_centered = ratings_matrix.sub(
    user_means,
    axis=0
).fillna(0)


# Obliczenie podobienstwa kosinusowego miedzy uzytkownikami
similarity_matrix = pd.DataFrame(
    cosine_similarity(ratings_centered),
    index=ratings_matrix.index,
    columns=ratings_matrix.index
)


def similar_users_collaborative(user, k=3):
    """
    Zwraca k najbardziej podobnych uzytkownikow.

    Parametry:
        user - nazwa uzytkownika
        k - liczba podobnych uzytkownikow

    Wynik:
        Lista slownikow z uzytkownikami i podobienstwem.
        None, jesli uzytkownik nie istnieje.
    """

    if user not in ratings_matrix.index:
        return None

    if not isinstance(k, int) or k < 1:
        raise ValueError("Parametr k musi byc liczba calkowita wieksza od 0.")

    similarities = similarity_matrix.loc[user].drop(user)

    # Sortowanie od najbardziej podobnych uzytkownikow
    similarities = similarities.sort_values(ascending=False)

    neighbors = similarities.head(k)

    return [
        {
            "user": neighbor,
            "similarity": float(similarity)
        }
        for neighbor, similarity in neighbors.items()
    ]


def recommend_collaborative(user, top_n=5, k=3):
    """
    Rekomenduje filmy uzytkownikowi na podstawie ocen
    wystawionych przez podobnych uzytkownikow.

    Parametry:
        user - nazwa uzytkownika
        top_n - liczba rekomendowanych filmow
        k - liczba podobnych uzytkownikow

    Wynik:
        Lista rekomendacji zawierajaca:
        - title: tytul filmu
        - predicted: przewidywana ocena
        - because: uzytkownicy, na podstawie ktorych
                   film zostal polecony

        None, jesli uzytkownik nie istnieje.
    """

    if user not in ratings_matrix.index:
        return None

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError(
            "Parametr top_n musi byc liczba calkowita wieksza od 0."
        )

    if not isinstance(k, int) or k < 1:
        raise ValueError(
            "Parametr k musi byc liczba calkowita wieksza od 0."
        )

    # Filmy, ktore uzytkownik juz ocenil
    rated_movies = ratings_matrix.loc[user].dropna().index

    # Filmy, ktorych uzytkownik jeszcze nie ocenil
    unrated_movies = ratings_matrix.columns.difference(rated_movies)

    # Pobranie najbardziej podobnych uzytkownikow
    neighbors = similar_users_collaborative(user, k)

    # Srednia ocena uzytkownika
    user_mean = float(user_means.loc[user])

    recommendations = []

    for movie in unrated_movies:

        weighted_sum = 0.0
        similarity_sum = 0.0
        because = []

        # Obliczamy przewidywana ocene na podstawie sasiadow
        for neighbor_info in neighbors:

            neighbor = neighbor_info["user"]
            similarity = neighbor_info["similarity"]

            # Pomijamy sasiadow, ktorzy nie ocenili danego filmu
            neighbor_rating = ratings_matrix.loc[neighbor, movie]

            if pd.isna(neighbor_rating):
                continue

            neighbor_mean = float(user_means.loc[neighbor])

            # Odchylenie oceny sasiada od jego sredniej
            rating_deviation = float(neighbor_rating) - neighbor_mean

            # Wazona suma odchylen
            weighted_sum += similarity * rating_deviation

            # Suma wartosci bezwzglednych podobienstw
            similarity_sum += abs(similarity)

            because.append(
                {
                    "user": neighbor,
                    "rating": float(neighbor_rating),
                    "similarity": float(similarity)
                }
            )

        # Jesli nie ma sasiadow z ocena filmu, pomijamy film
        if similarity_sum == 0:
            continue

        # Przewidywana ocena:
        # srednia uzytkownika + wazona suma odchylen
        predicted_rating = user_mean + (
            weighted_sum / similarity_sum
        )

        # Ograniczenie przewidywanej oceny do zakresu 1-5
        predicted_rating = max(
            1.0,
            min(5.0, predicted_rating)
        )

        recommendations.append(
            {
                "title": movie,
                "predicted": round(float(predicted_rating), 2),
                "because": because
            }
        )

    # Sortowanie od najwyzszej przewidywanej oceny
    recommendations.sort(
        key=lambda recommendation: recommendation["predicted"],
        reverse=True
    )

    # Zwracamy top_n najlepszych rekomendacji
    return recommendations[:top_n]

if __name__ == "__main__":
    print("Wymiary danych:")
    print(df.shape)

    print("\nPierwsze 3 wiersze:")
    print(df.head(3))

    print("\nBrakujące wartości:")
    print(df.isnull().sum())

    print("\nWymiar macierzy TF-IDF:")
    print(tfidf_matrix.shape)

    print("\nRekomendacje dla Inception:")
    print(recommend("Inception"))

    print("\nRekomendacje kolaboratywne dla Ani:")
    print(recommend_collaborative("Ania", top_n=5, k=3))
