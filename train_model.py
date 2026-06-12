import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv('movies.csv')

# Create vectors from genres
cv = CountVectorizer()
vectors = cv.fit_transform(movies['genre']).toarray()

# Compute similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]

    print("Recommended Movies:")
    for i in movies_list:
        print(movies.iloc[i[0]].title)

# Test
recommend("Avatar")