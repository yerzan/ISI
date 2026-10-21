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
