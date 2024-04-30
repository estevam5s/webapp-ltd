# ChatBot Starter com Streamlit, OpenAI e LangChain

Este repositório contém um chatbot simples, mas poderoso, desenvolvido com Streamlit, OpenAI e LangChain. O chatbot mantém a memória conversacional, o que significa que pode fazer referência a trocas passadas nas suas respostas.

## Visão geral

O chatbot é uma demonstração da integração do modelo GPT da OpenAI, da biblioteca LangChain e do Streamlit para a criação de aplicativos web interativos. A memória conversacional do bot permite manter o contexto durante a sessão de chat, levando a uma experiência de usuário mais coerente e envolvente. É importante ressaltar que este aplicativo chatbot rico em recursos é implementado em menos de 40 linhas de código (excluindo espaços em branco e comentários)!

### Características principais

- **Streamlit:** Uma estrutura Python poderosa e rápida usada para criar a interface web para o chatbot.
- **GPT da OpenAI:** Um modelo de IA de processamento de linguagem de última geração que gera as respostas do chatbot.
- **LangChain:** Uma biblioteca wrapper para o modelo ChatGPT que ajuda a gerenciar o histórico de conversas e estruturar as respostas do modelo.
 
## Aplicativo de demonstração

[![Aplicativo Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://)

## Como correr

### Pré-requisitos

- Python 3.6 ou superior
- Streamlit
- LangChain
- Chave API OpenAI

### Passos

1. Clone este repositório.
2. Instale os pacotes Python necessários usando o comando `pip install -r requirements.txt`.
3. Defina a variável de ambiente para sua chave de API OpenAI.
4. Execute o aplicativo Streamlit usando o comando `streamlit run app.py`.

## Uso

O chatbot começa com uma mensagem do sistema que dá o tom da conversa. Em seguida, ele alterna entre receber informações do usuário e gerar respostas de IA. O histórico da conversa é armazenado e utilizado como contexto para geração de respostas futuras, permitindo ao chatbot manter a continuidade da conversação.

## Contribuição

Contribuições, problemas e solicitações de recursos são bem-vindos.

## Licença

Este projeto está licenciado sob os termos da licença do MIT.
