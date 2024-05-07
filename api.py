import streamlit as st
import requests

def technology_news_page():
  # Define a função para obter as notícias sobre inteligência artificial e linguagens de programação da API
  def get_ai_news():
      try:
          url = "https://newsapi.org/v2/everything"
          params = {
              "q": "artificial intelligence OR programming language",
              "apiKey": "adefb2b75dd2463b82c7611c86c09b74"  # Substitua "SUA_CHAVE_DE_API_AQUI" pela sua chave de API
          }
          response = requests.get(url, params=params)
          data = response.json()

          # Verifica se há erro na resposta
          if response.status_code != 200 or data.get("status") != "ok":
              st.error("Erro ao recuperar as notícias. Por favor, tente novamente mais tarde.")
              return None

          articles = data.get("articles", [])

          # Verifica se há notícias disponíveis
          if articles:
              return articles
          else:
              st.error("Nenhuma notícia encontrada sobre inteligência artificial ou linguagens de programação.")
              return None
      except Exception as e:
          st.error(f"Ocorreu um erro: {e}")
          return None

  # Define o título do aplicativo
  st.title("Notícias sobre Inteligência Artificial e Linguagens de Programação")

  # Obtém as notícias sobre inteligência artificial e linguagens de programação da API
  ai_news = get_ai_news()

  # Se houver notícias disponíveis, exibe cada uma delas
  if ai_news:
      for idx, article in enumerate(ai_news, start=1):
          st.subheader(f"Notícia {idx}")
          st.write(article["title"])
          st.write(article["description"])
          if article["urlToImage"]:
              st.image(article["urlToImage"], caption="Imagem da Notícia")
          st.write("Fonte:", article["source"]["name"])
          st.write("Autor:", article.get("author", "N/A"))
          st.write("Publicado em:", article["publishedAt"])
          st.write("Leia mais:", article["url"])
          st.write("---")  # Adiciona uma linha horizontal entre cada notícia
