from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and data
model_filename = 'models/knn_model.joblib'
tfidf_matrix_filename = 'models/tfidf_matrix.joblib'
indices_filename = 'models/indices.joblib'
df_movies_filename = 'data/df_movies.csv'

loaded_nn_model = joblib.load(model_filename)
loaded_tfidf_matrix = joblib.load(tfidf_matrix_filename)
loaded_indices = joblib.load(indices_filename)
df_movies = pd.read_csv(df_movies_filename)

def get_recommendations(title: str):
    if title not in loaded_indices:
        return None, f"Erro: Filme '{title}' não encontrado no dataset."
    idx = loaded_indices[title]
    movie_vector = loaded_tfidf_matrix[idx]
    distances, movie_indices = loaded_nn_model.kneighbors(movie_vector)
    similar_movie_indices = movie_indices.flatten()[1:]
    recommended_movies = df_movies.iloc[similar_movie_indices]
    return recommended_movies[['title', 'genre', 'rating', 'description']], None

HTML_FORM = '''
<!doctype html>
<title>Movie Recommender</title>
<h1>Movie Recommender</h1>
<form method="post">
  <label for="title">Movie Title:</label>
  <input type="text" id="title" name="title" required>
  <input type="submit" value="Recommend">
</form>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
{% if recommendations is not none %}
  <h2>Recommended Movies{% if typed_title %} for "{{ typed_title }}"{% endif %}:</h2>
  <ul>
  {% for _, row in recommendations.iterrows() %}
    <li><b>{{ row['title'] }}</b> ({{ row['genre'] }}, Rating: {{ row['rating'] }})<br>{{ row['description'] }}</li>
  {% endfor %}
  </ul>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    error = None
    typed_title = None
    if request.method == 'POST':
        title = request.form['title'].strip()
        typed_title = title
        recs, error = get_recommendations(title)
        if recs is not None:
            recommendations = recs
    return render_template_string(HTML_FORM, recommendations=recommendations, error=error, typed_title=typed_title)

if __name__ == '__main__':
    app.run(debug=True) 