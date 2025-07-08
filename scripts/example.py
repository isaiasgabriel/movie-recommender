import joblib
import pandas as pd

# Define os nomes dos arquivos salvos
model_filename = 'models/knn_model.joblib'
tfidf_matrix_filename = 'models/tfidf_matrix.joblib'
indices_filename = 'models/indices.joblib'
df_movies_filename = 'data/df_movies.csv'

print("\nCarregando o modelo KNN, matriz TF-IDF, índices e DataFrame de filmes...")
# Carrega o modelo KNN
loaded_nn_model = joblib.load(model_filename)
print("Modelo KNN carregado com sucesso.")

# Carrega a matriz TF-IDF
loaded_tfidf_matrix = joblib.load(tfidf_matrix_filename)
print("Matriz TF-IDF carregada com sucesso.")

# Carrega os índices
loaded_indices = joblib.load(indices_filename)
print("Índices carregados com sucesso.")

# Carrega o DataFrame de filmes
df_movies = pd.read_csv(df_movies_filename)
print("DataFrame de filmes carregado com sucesso.")

def get_recommendations_from_loaded(title: str):
    """
    Função que recebe o título de um filme e retorna 5 filmes similares
    usando os objetos carregados.
    """
    # 1. Obter o índice do filme a partir do seu título usando os índices carregados
    if title not in loaded_indices:
        return f"Erro: Filme '{title}' não encontrado no dataset carregado."

    idx = loaded_indices[title]

    # 2. Obter o vetor TF-IDF do filme selecionado da matriz carregada
    movie_vector = loaded_tfidf_matrix[idx]

    # 3. Usar o modelo KNN carregado para encontrar os 6 filmes mais próximos
    distances, movie_indices = loaded_nn_model.kneighbors(movie_vector)

    # 4. Pegar os índices dos 5 filmes mais similares (ignorando o primeiro)
    similar_movie_indices = movie_indices.flatten()[1:]

    # 5. Retornar os títulos e detalhes dos filmes recomendados
    recommended_movies = df_movies.iloc[similar_movie_indices]

    return recommended_movies[['title', 'genre', 'description']]

print("\n--- TESTANDO O SISTEMA DE RECOMENDAÇÃO CARREGADO ---")

# Exemplo de teste com o modelo carregado
movie_choice_loaded = "The Dark Knight" # Use um filme que sabemos que está no dataset
print(f"\nRecomendações (carregadas) para '{movie_choice_loaded}':")
recommendations_loaded = get_recommendations_from_loaded(movie_choice_loaded)
print(recommendations_loaded)

# Exemplo de teste com um filme que não existe (no dataset carregado)
movie_choice_loaded_fail = "Um Filme Inexistente Para Teste"
print(f"\nRecomendações (carregadas) para '{movie_choice_loaded_fail}':")
recommendations_loaded_fail = get_recommendations_from_loaded(movie_choice_loaded_fail)
print(recommendations_loaded_fail)