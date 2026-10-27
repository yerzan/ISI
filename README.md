
# System rekomendacji filmów — ISI

## Opis projektu

Projekt realizowany w ramach laboratoriów z przedmiotu Inteligentne Systemy Internetowe (ISI).

Aplikacja umożliwia rekomendowanie filmów na podstawie danych o filmach oraz ocen użytkowników. Projekt rozwijany jest etapami, z zachowaniem funkcjonalności z poprzednich laboratoriów.

W projekcie wykorzystano:
- Python
- Flask
- Pandas
- Scikit-learn
- pytest

## Funkcjonalności

### Lab 1 — API filmów

Podstawowa aplikacja Flask udostępniająca API związane z filmami.

### Lab 2 — Rekomendacje content-based

Rekomendacje filmów na podstawie ich cech tekstowych.

Wykorzystano:
- TF-IDF do reprezentacji opisów filmów,
- podobieństwo cosinusowe do porównywania filmów,
- rekomendowanie filmów podobnych do wskazanego tytułu.

Dane filmowe znajdują się w pliku `movies.csv`.

### Lab 3 — Rekomendacje kolaboratywne

Rekomendacje filmów na podstawie ocen użytkowników.

Wykorzystano:
- macierz użytkownik–film,
- centrowanie ocen względem średniej użytkownika,
- podobieństwo cosinusowe użytkowników,
- przewidywanie ocen na podstawie ocen podobnych użytkowników.

Dane o ocenach znajdują się w pliku `ratings.csv`.

Rekomendacje pomijają filmy, które użytkownik już ocenił.

## Struktura projektu

```text
ISI/
├── app.py
├── recommender.py
├── movies.csv
├── ratings.csv
├── test_lab03.py
├── README.md
└── requirements.txt
```


## Instalacja

Projekt wykorzystuje Python oraz biblioteki zapisane w pliku `requirements.txt`.

Zalecane jest uruchomienie aplikacji w wirtualnym środowisku.

Utworzenie środowiska:

```powershell
python -m venv .venv
```

Aktywacja środowiska w PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalacja zależności projektu:

```powershell
python -m pip install -r requirements.txt
```

## Uruchomienie aplikacji

Po zainstalowaniu zależności uruchom aplikację:

```powershell
python app.py
```

Aplikacja będzie dostępna pod adresem:

```text
http://127.0.0.1:5000
```

## Testowanie

Uruchom testy Lab 3:

```powershell
python -m pytest test_lab03.py -q
```

Oczekiwany wynik po pomyślnym wykonaniu testów:

```text
13 passed
```

## API — rekomendacje kolaboratywne

Endpoint:

```text
GET /api/recommend/collaborative
```

Wymagany parametr:

| Parametr | Opis |
|---|---|
| `user` | Nazwa użytkownika, dla którego generowane są rekomendacje |

Opcjonalne parametry:

| Parametr | Domyślna wartość | Opis |
|---|---:|---|
| `top_n` | 5 | Liczba rekomendowanych filmów |
| `k` | 3 | Liczba podobnych użytkowników uwzględnianych w rekomendacji |

Przykładowe zapytanie:

```text
http://127.0.0.1:5000/api/recommend/collaborative?user=Ania&top_n=5&k=3
```

Odpowiedź zawiera:
- `user` — użytkownika, dla którego przygotowano rekomendacje,
- `recommendations` — listę rekomendowanych filmów,
- `title` — tytuł filmu,
- `predicted` — przewidywaną ocenę,
- `because` — uzasadnienie rekomendacji na podstawie ocen podobnych użytkowników.
