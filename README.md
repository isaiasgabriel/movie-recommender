## Movie Recommender

A simple movie recommender system using K-Nearest Neighbors.

### Model Development

The recommendation model was developed using a K-Nearest Neighbors (KNN) algorithm. The process is detailed in the `notebooks/training-model-code.ipynb` notebook and can be summarized in the following steps:

1.  **Data Loading**: The script downloads the "[IMDb Movies Dataset](https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre)" from Kaggle.
2.  **Preprocessing**:
    *   Relevant columns (`movie_name`, `genre`, `description`, `rating`) are selected.
    *   The `movie_name` column is renamed to `title`.
    *   A new `content` column is created by combining the `description` and `genre` of each movie.
3.  **Feature Extraction**: The `TfidfVectorizer` is used to convert the textual data in the `content` column into a numerical TF-IDF matrix.
4.  **Model Training**: A `NearestNeighbors` model is "trained" on the TF-IDF matrix using cosine similarity to find the most similar movies.
5.  **Artifacts Generation**: The script saves the trained KNN model, the TF-IDF matrix, and a title-to-index mapping as `.joblib` files to be used by the Flask application. 

### Project Structure

```
.
├── app.py
├── data
│   └── df_movies.csv
├── models
│   ├── indices.joblib
│   ├── knn_model.joblib
│   └── tfidf_matrix.joblib
├── notebooks
│   └── training-model-code.ipynb
└── scripts
    └── example.py
```

### Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:isaiasgabriel/movie-recommender.git
   ```
2. Navigate to the project directory:
   ```bash
   cd movie-recommender
   ```
3. Install the required dependencies:
   ```bash
   pip install flask joblib pandas
   ```

### Usage

To run the application, execute the following command in the root directory:

```bash
python app.py
```

Then, open your web browser and go to `http://127.0.0.1:5000/`.
