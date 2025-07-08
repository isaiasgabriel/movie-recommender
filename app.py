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
        return None, f"Error: Movie '{title}' not found in the dataset."
    idx = loaded_indices[title]
    movie_vector = loaded_tfidf_matrix[idx]
    distances, movie_indices = loaded_nn_model.kneighbors(movie_vector)
    similar_movie_indices = movie_indices.flatten()[1:]
    recommended_movies = df_movies.iloc[similar_movie_indices]
    return recommended_movies[['title', 'genre', 'description']], None

HTML_FORM = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Movie Recommender</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: #f4f4f9;
      color: #333;
      margin: 0;
      padding: 2rem;
      display: flex;
      justify-content: center;
      align-items: center;
      flex-direction: column;
    }
    .container {
      background: white;
      padding: 2rem;
      border-radius: 8px;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
      width: 100%;
      max-width: 800px;
    }
    h1 {
      color: #5a67d8;
      text-align: center;
    }
    form {
      display: flex;
      gap: 1rem;
      margin-bottom: 2rem;
    }
    input[type="text"] {
      flex-grow: 1;
      padding: 0.75rem;
      border: 1px solid #ddd;
      border-radius: 4px;
      font-size: 1rem;
    }
    input[type="submit"] {
      padding: 0.75rem 1.5rem;
      background-color: #5a67d8;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 1rem;
      transition: background-color 0.2s;
    }
    input[type="submit"]:hover {
      background-color: #434190;
    }
    .error {
      color: #e53e3e;
      text-align: center;
      margin-bottom: 1rem;
    }
    .recommendations {
      list-style: none;
      padding: 0;
    }
    .recommendation-item {
      background-color: #f9f9f9;
      border: 1px solid #eee;
      padding: 1rem;
      margin-bottom: 1rem;
      border-radius: 4px;
      transition: box-shadow 0.2s;
    }
    .recommendation-item:hover {
      box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .recommendation-item h3 {
      margin: 0 0 0.5rem 0;
      color: #434190;
    }
    .recommendation-item p {
      margin: 0;
    }
    .meta {
      font-size: 0.9rem;
      color: #666;
      margin-bottom: 0.5rem;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Movie Recommender</h1>
    <form method="post">
      <input type="text" id="title" name="title" placeholder="Enter a movie title..." required>
      <input type="submit" value="Recommend">
    </form>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    {% if recommendations is not none %}
      <h2>Recommended movies{% if typed_title %} for "{{ typed_title }}"{% endif %}:</h2>
      <ul class="recommendations">
      {% for _, row in recommendations.iterrows() %}
        <li class="recommendation-item">
          <h3>{{ row['title'] }}</h3>
          <p class="meta"><b>Genre:</b> {{ row['genre'] }}</p>
          <p>{{ row['description'] }}</p>
        </li>
      {% endfor %}
      </ul>
    {% endif %}
  </div>
</body>
</html>
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