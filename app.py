import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load data
columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
           'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
           'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
           'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, names=columns)

# Combine genres
genre_columns = columns[5:]
movies['genres'] = movies[genre_columns].apply(lambda row: ' '.join([genre for genre, val in row.items() if val == 1]), axis=1)

# Drop extras
movies = movies[['movie_id', 'title', 'genres']]
movies['content'] = movies['title'] + ' ' + movies['genres']

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def recommend_movies(title, cosine_sim=cosine_sim):
    if title not in indices:
        return ["Movie not found in database."]
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    
    return movies['title'].iloc[movie_indices].tolist()


# ----------------- Streamlit UI -----------------
st.title("🎬 Movie Recommender System")

movie_name = st.text_input("Enter the movie name (e.g., Toy Story)").strip()
release_year = st.text_input("Enter the release year (e.g., 1995)").strip()

if st.button("Recommend"):
    full_title = f"{movie_name} ({release_year})"

    # Case-insensitive title match
    all_titles = [title.lower() for title in movies['title']]
    if full_title.lower() in all_titles:
        # Find the exact title as it appears in dataset
        matched_title = movies['title'][all_titles.index(full_title.lower())]
        recommendations = recommend_movies(matched_title)

        st.subheader("Top 5 Similar Movies:")
        for movie in recommendations:
            st.write("👉", movie)
    else:
        st.warning("❌ Movie not found in the dataset. Please check the title and year.")

