import joblib
import pandas as pd

# Define os nomes dos arquivos salvos
model_filename = 'knn_model.joblib'
tfidf_matrix_filename = 'tfidf_matrix.joblib'
indices_filename = 'indices.joblib'

print("\nCarregando o modelo KNN, matriz TF-IDF e índices dos arquivos...")
# Carrega o modelo KNN
loaded_nn_model = joblib.load(model_filename)
print("Modelo KNN carregado com sucesso.")

# Carrega a matriz TF-IDF
loaded_tfidf_matrix = joblib.load(tfidf_matrix_filename)
print("Matriz TF-IDF carregada com sucesso.")

# Carrega os índices
loaded_indices = joblib.load(indices_filename)
print("Índices carregados com sucesso.")

# IMPORTANTE: Para usar a função get_recommendations, você também precisará do df_movies
# Você pode salvar o df_movies em um arquivo (por exemplo, CSV ou pickle) e carregá-lo aqui,
# OU recriar o df_movies a partir dos dados originais se for rápido o suficiente.
# Para este exemplo, vamos assumir que df_movies já foi carregado ou recriado.
# Se df_movies não existir, a próxima linha falhará.

# Recriando uma versão mínima de df_movies apenas com as colunas necessárias para a função
# Em um cenário real, você carregaria o df_movies completo ou um subconjunto relevante.
# Para este exemplo de carregamento, vamos apenas criar um DataFrame dummy que se pareça com df_movies
# para que a função de recomendação possa ser testada.
# **Em um caso de uso real, você carregaria o df_movies salvo.**
# Exemplo: loaded_df_movies = pd.read_pickle('df_movies.pkl')

# Para o propósito de DEMONSTRAR o carregamento:
# Precisamos do DataFrame original (ou pelo menos as colunas 'title', 'genre', 'rating', 'description')
# para que a função get_recommendations possa buscar os detalhes dos filmes recomendados.
# Se você não salvou o df_movies, você precisará carregá-lo ou recriá-lo.
# Vamos usar o df_movies que já está na sessão para este teste.
# Em um cenário "do zero" (novo notebook), você carregaria o df_movies salvo.

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

    # 5. Retornar os títulos e detalhes dos filmes recomendados usando o df_movies original (ou carregado)
    # **Importante:** Nesta demonstração, estamos usando o 'df_movies' da sessão atual.
    # Em um cenário real de carregamento, você usaria um 'loaded_df_movies'.
    recommended_movies = df_movies.iloc[similar_movie_indices] # Use df_movies da sessão atual

    return recommended_movies[['title', 'genre', 'rating', 'description']]


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