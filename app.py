# Import required libraries
from dotenv import load_dotenv
import requests
import qrcode
from io import BytesIO
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from openpyxl import load_workbook  # Importe a função load_workbook aqui
from itertools import zip_longest
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
# from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
from pages.digital import curriculoVintage
import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

def chatbotGemeni():
    load_dotenv()
    GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)

    new_chat_id = f'{time.time()}'
    MODEL_ROLE = 'ai'
    AI_AVATAR_ICON = '✨'

    # Create a data/ folder if it doesn't already exist
    try:
        os.mkdir('data/')
    except:
        # data/ folder already exists
        pass

    # Load past chats (if available)
    try:
        past_chats: dict = joblib.load('data/past_chats_list')
    except:
        past_chats = {}

    # Sidebar allows a list of past chats
    with st.sidebar:
        st.write('# Past Chats')
        if st.session_state.get('chat_id') is None:
            st.session_state.chat_id = st.selectbox(
                label='Pick a past chat',
                options=[new_chat_id] + list(past_chats.keys()),
                format_func=lambda x: past_chats.get(x, 'New Chat'),
                placeholder='_',
            )
        else:
            # This will happen the first time AI response comes in
            st.session_state.chat_id = st.selectbox(
                label='Pick a past chat',
                options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
                index=1,
                format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
                placeholder='_',
            )
        # Save new chats after a message has been sent to AI
        # TODO: Give user a chance to name chat
        st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

    st.write('# Chat with Gemini')

    # Chat history (allows to ask multiple questions)
    try:
        st.session_state.messages = joblib.load(
            f'data/{st.session_state.chat_id}-st_messages'
        )
        st.session_state.gemini_history = joblib.load(
            f'data/{st.session_state.chat_id}-gemini_messages'
        )
        print('old cache')
    except:
        st.session_state.messages = []
        st.session_state.gemini_history = []
        print('new_cache made')
    st.session_state.model = genai.GenerativeModel('gemini-pro')
    st.session_state.chat = st.session_state.model.start_chat(
        history=st.session_state.gemini_history,
    )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(
            name=message['role'],
            avatar=message.get('avatar'),
        ):
            st.markdown(message['content'])

    # React to user input
    if prompt := st.chat_input('Your message here...'):
        # Save this as a chat for later
        if st.session_state.chat_id not in past_chats.keys():
            past_chats[st.session_state.chat_id] = st.session_state.chat_title
            joblib.dump(past_chats, 'data/past_chats_list')
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(
            dict(
                role='user',
                content=prompt,
            )
        )
        ## Send message to AI
        response = st.session_state.chat.send_message(
            prompt,
            stream=True,
        )
        # Display assistant response in chat message container
        with st.chat_message(
            name=MODEL_ROLE,
            avatar=AI_AVATAR_ICON,
        ):
            message_placeholder = st.empty()
            full_response = ''
            assistant_response = response
            # Streams in a chunk at a time
            for chunk in response:
                # Simulate stream of chunk
                # TODO: Chunk missing `text` if API stops mid-stream ("safety"?)
                for ch in chunk.text.split(' '):
                    full_response += ch + ' '
                    time.sleep(0.05)
                    # Rewrites with a cursor at end
                    message_placeholder.write(full_response + '▌')
            # Write full message with placeholder
            message_placeholder.write(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            dict(
                role=MODEL_ROLE,
                content=st.session_state.chat.history[-1].parts[0].text,
                avatar=AI_AVATAR_ICON,
            )
        )
        st.session_state.gemini_history = st.session_state.chat.history
        # Save to file
        joblib.dump(
            st.session_state.messages,
            f'data/{st.session_state.chat_id}-st_messages',
        )
        joblib.dump(
            st.session_state.gemini_history,
            f'data/{st.session_state.chat_id}-gemini_messages',
        )

# Load environment variables
load_dotenv()
st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

#TODO Page - chatbot
# ------------------------------------------------------------------------------
# Initialize session state variables for chatbot
def init_chatbot_session_state():
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []  # Store AI generated responses

    if 'past' not in st.session_state:
        st.session_state['past'] = []  # Store past user inputs

    if 'entered_prompt' not in st.session_state:
        st.session_state['entered_prompt'] = ""  # Store the latest user input

# Initialize session state variables for resume generator
def init_resume_session_state():
    if 'name' not in st.session_state:
        st.session_state['name'] = ""
    if 'email' not in st.session_state:
        st.session_state['email'] = ""
    if 'phone' not in st.session_state:
        st.session_state['phone'] = ""
    if 'experience' not in st.session_state:
        st.session_state['experience'] = ""
    if 'education' not in st.session_state:
        st.session_state['education'] = ""
    if 'skills' not in st.session_state:
        st.session_state['skills'] = ""

# Initialize the ChatOpenAI model
def init_chatbot_model(api_key):
    return ChatOpenAI(
        temperature=0.5,
        model_name="gpt-3.5-turbo",
        api_key="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"
    )

# Build a list of messages including system, human and AI messages for chatbot
def build_message_list():
    zipped_messages = [SystemMessage(
        content="You are a helpful AI assistant talking with a human. If you do not know an answer, just say 'I don't know', do not make up an answer.")]

    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg))  
        if ai_msg is not None:
            zipped_messages.append(
                AIMessage(content=ai_msg))  

    return zipped_messages

# Generate AI response using the ChatOpenAI model
def generate_response(chat):
    zipped_messages = build_message_list()
    ai_response = chat(zipped_messages)
    return ai_response.content

# Initialize the ChatOpenAI model
def init_chat():
    return ChatOpenAI(
        temperature=0.5,
        model_name="gpt-3.5-turbo"
    )

# Initialize session state variables
def init_session_state():
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []  # Store AI generated responses

    if 'past' not in st.session_state:
        st.session_state['past'] = []  # Store past user inputs

    if 'entered_prompt' not in st.session_state:
        st.session_state['entered_prompt'] = ""  # Store the latest user input


# Function for the chatbot page
def chatbot_page():
    st.title("ChatBot LTD")

    # Initialize session state variables
    init_session_state()

    # Initialize the ChatOpenAI model
    chat = init_chat()

    # Create a text input for user
    user_input = st.text_input('YOU: ', key='prompt_input')

    if st.button("Enviar"):
        st.session_state.entered_prompt = user_input

    if st.session_state.entered_prompt != "":
        # Get user query
        user_query = st.session_state.entered_prompt

        # Append user query to past queries
        st.session_state.past.append(user_query)

        # Generate response
        output = generate_response(chat)

        # Append AI response to generated responses
        st.session_state.generated.append(output)

    # Display the chat history
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            # Display AI response
            message(st.session_state["generated"][i], key=str(i))
            # Display user message
            message(st.session_state['past'][i],
                    is_user=True, key=str(i) + '_user')

    st.markdown("""
    ---
    Feito por [Estevam Souza](https://github.com/estevam5s)""")

#TODO Page - About
# ------------------------------------------------------------------------------
# Function for the "About" page
def about_page():
    st.markdown("""
    # Bem-vindo ao ChatBot LTD!

    Este é um aplicativo que inclui um chatbot baseado em inteligência artificial e um gerador de currículo em PDF.

    ## ChatBot
    O ChatBot LTD é um assistente de IA projetado para ajudar os usuários a obter respostas para suas perguntas. Ele usa o modelo GPT-3.5 para gerar respostas com base nos inputs dos usuários.

    ## Gerador de Currículo
    O Gerador de Currículo permite que os usuários criem rapidamente um currículo profissional em formato PDF. Basta preencher as informações pessoais, experiência profissional, educação e habilidades, e o currículo será gerado automaticamente em PDF para download.

    ## Organização no GitHub
    Para mais informações sobre nossa organização no GitHub, confira [aqui](https://github.com/suaorganizacao).

    ---    
    Feito com ❤️ por [Estevam Souza](https://github.com/estevam5s)""")

#TODO Page - AI Tools
# ------------------------------------------------------------------------------
# Function for the "AI Tools" page
def about_page_ia():
    st.title("Sobre")
    st.markdown("""
    Este aplicativo foi criado para fornecer informações sobre o mercado de trabalho relacionado à Inteligência Artificial (IA) e destacar diversas plataformas e serviços que utilizam IA.
    
    ## Objetivo
    
    O objetivo deste aplicativo é ajudar os usuários a entenderem melhor as oportunidades de carreira em IA e descobrir ferramentas e recursos úteis disponíveis na área.
    
    ## Desenvolvimento
    
    Este aplicativo foi desenvolvido utilizando Streamlit, uma biblioteca do Python para criação de aplicativos da web de maneira rápida e fácil.
    
    ## Fontes
    
    As informações fornecidas neste aplicativo são baseadas em diversas fontes, incluindo artigos, sites especializados e recomendações de profissionais da área de IA.
    """)


def ai_careers_page():
    st.title("Carreiras com IA")
    st.markdown("""
    Existem diversas oportunidades de carreira em Inteligência Artificial (IA), que vão desde desenvolvedores de software especializados em IA até cientistas de dados e engenheiros de machine learning. Aqui estão algumas das principais carreiras em IA:
    
    ### Desenvolvedor de Software de IA
    Os desenvolvedores de software de IA são responsáveis por projetar, desenvolver e implementar sistemas de IA, incluindo algoritmos de aprendizado de máquina e redes neurais.
    
    ### Cientista de Dados
    Os cientistas de dados utilizam técnicas de IA para analisar grandes volumes de dados e extrair insights valiosos que ajudam as empresas a tomar decisões informadas.
    
    ### Engenheiro de Machine Learning
    Os engenheiros de machine learning trabalham no desenvolvimento e otimização de modelos de IA, criando sistemas capazes de aprender e melhorar com o tempo.
    
    ### Especialista em Processamento de Linguagem Natural (NLP)
    Os especialistas em NLP desenvolvem sistemas de IA capazes de compreender e gerar linguagem humana, permitindo aplicações como assistentes virtuais e tradução automática.
    """)


def ai_tools_page():
    st.title("Inteligência Artificial no Mercado de Trabalho")
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title('Navegação')
    page = st.sidebar.radio('Ir para:', ('Sobre', 'Carreiras com IA', 'Plataformas de Criação de Currículos com IA', 'Plataformas que Utilizam Inteligência Artificial', 'TheresaNaiForThat', 'Modelos de Linguagem Semelhantes ao ChatGPT'))

    if page == 'Sobre':
        about_page_ia()

    elif page == 'Carreiras com IA':
        ai_careers_page()

    elif page == 'Plataformas de Criação de Currículos com IA':
        st.header("Plataformas de Criação de Currículos com IA")
        st.markdown("Aqui estão algumas plataformas que utilizam Inteligência Artificial para criar currículos:")
        st.markdown("- [Enhancv](https://www.enhancv.com/)")
        st.markdown("- [Kickresume](https://www.kickresume.com/)")
        st.markdown("- [Resumai](https://www.resumai.com/)")
        st.markdown("- [RX Resume](https://rxresu.me/)")
        st.markdown("- [Resumaker](https://resumaker.ai/)")
        st.markdown("- [Zety](https://zety.com/br/gerador-de-curriculo)")

    elif page == 'Plataformas que Utilizam Inteligência Artificial':
        st.header("Plataformas que Utilizam Inteligência Artificial")
        st.markdown("Além das plataformas de criação de currículos, há diversas outras que utilizam Inteligência Artificial para oferecer serviços úteis:")
        st.markdown("- [GetLazy](https://www.getlazy.ai/)")
        st.markdown("- [Geospy](https://geospy.ai/)")
        st.markdown("- [Teal](https://www.tealhq.com/)")
        st.markdown("- [OnlineCV](https://www.onlinecurriculo.com/)")
        st.markdown("- [OpenInterpreter](https://github.com/OpenInterpreter/open-interpreter)")

    elif page == 'TheresaNaiForThat':
        st.header("TheresaNaiForThat")
        st.markdown("O [TheresaNaiForThat](https://theresanaiforthat.com/) é uma plataforma que compila e encontra diversos serviços que usam Inteligência Artificial (IA).")
        st.markdown("O site apresenta milhares de ferramentas disponíveis na Internet, divididas entre mais de 600 categorias.")
        st.markdown("Fonte: [TechTudo](https://www.techtudo.com.br/dicas-e-tutoriais/2023/03/there-is-an-ai-for-that-saiba-usar-site-que-encontra-gratis-ias-uteis-edsoftwares.ghtml)")

    elif page == 'Modelos de Linguagem Semelhantes ao ChatGPT':
        st.header("Modelos de Linguagem Semelhantes ao ChatGPT")
        st.markdown("Aqui estão alguns sites que oferecem modelos de linguagem semelhantes ao ChatGPT:")
        st.markdown("- [Perplexity](https://www.perplexity.com/)")
        st.markdown("- [Gemeni](https://gemeni.ai/)")
        st.markdown("- [Black Box](https://www.blackbox.com/)")
        st.markdown("- [EleutherAI](https://www.eleuther.ai/)")
        st.markdown("- [OpenAI](https://openai.com/)")
        st.markdown("- [Hugging Face](https://huggingface.co/)")
        st.markdown("- [DeepAI](https://deepai.org/)")
        st.markdown("- [EleutherAI](https://www.eleuther.ai/)")

#TODO Page - IT Jobs
# ------------------------------------------------------------------------------
# Function for the IT Jobs page
def it_jobs_page():
    st.title("Vagas de Emprego em TI")

    st.markdown("""
    Aqui estão algumas das vagas de emprego disponíveis na área de Tecnologia da Informação:

    ### Desenvolvedor de Software
    - Descrição: Desenvolver e manter software para atender às necessidades da empresa.
    - Requisitos: Conhecimento em linguagens de programação como Python, Java, ou JavaScript.
    - [Vagas no Linkedin](https://www.linkedin.com/jobs/desenvolvedor-de-software)
    - [Vagas no Infojobs](https://www.infojobs.com.br/vagas-de-emprego-desenvolvedor%20de%20software.aspx)
    - [Vagas no Glassdoor](https://www.glassdoor.com.br/Vagas/desenvolvedor-de-software-vagas-SRCH_KO0,24.htm)

    ### Engenheiro de Dados
    - Descrição: Projetar e implementar sistemas de armazenamento e recuperação de dados.
    - Requisitos: Experiência em bancos de dados SQL e NoSQL, conhecimento em ferramentas de big data como Hadoop.
    - [Vagas no Linkedin](https://www.linkedin.com/jobs/engenheiro-de-dados)
    - [Vagas no Infojobs](https://www.infojobs.com.br/vagas-de-emprego-engenheiro%20de%20dados.aspx)
    - [Vagas no Glassdoor](https://www.glassdoor.com.br/Vagas/engenheiro-de-dados-vagas-SRCH_KO0,19.htm)

    ### Cientista de Dados
    - Descrição: Analisar grandes conjuntos de dados para extrair insights e tomar decisões baseadas em dados.
    - Requisitos: Habilidades em estatística, machine learning, e programação.
    - [Vagas no Linkedin](https://www.linkedin.com/jobs/cientista-de-dados)
    - [Vagas no Infojobs](https://www.infojobs.com.br/vagas-de-emprego-cientista%20de%20dados.aspx)
    - [Vagas no Glassdoor](https://www.glassdoor.com.br/Vagas/cientista-de-dados-vagas-SRCH_KO0,17.htm)

    ### Analista de Segurança da Informação
    - Descrição: Proteger os sistemas de informação da empresa contra ameaças internas e externas.
    - Requisitos: Conhecimento em segurança da informação, certificações como CISSP ou CompTIA Security+ são desejáveis.
    - [Vagas no Linkedin](https://www.linkedin.com/jobs/analista-de-seguran%C3%A7a-da-informa%C3%A7%C3%A3o)
    - [Vagas no Infojobs](https://www.infojobs.com.br/vagas-de-emprego-analista%20de%20seguran%C3%A7a%20da%20informa%C3%A7%C3%A3o.aspx)
    - [Vagas no Glassdoor](https://www.glassdoor.com.br/Vagas/analista-de-seguran%C3%A7a-da-informa%C3%A7%C3%A3o-vagas-SRCH_KO0,32.htm)

    ### Administrador de Redes
    - Descrição: Gerenciar e manter a infraestrutura de rede da empresa.
    - Requisitos: Experiência em administração de redes, conhecimento em protocolos de rede como TCP/IP.
    - [Vagas no Linkedin](https://www.linkedin.com/jobs/administrador-de-redes)
    - [Vagas no Infojobs](https://www.infojobs.com.br/vagas-de-emprego-administrador%20de%20redes.aspx)
    - [Vagas no Glassdoor](https://www.glassdoor.com.br/Vagas/administrador-de-redes-vagas-SRCH_KO0,22.htm)

    ## Roadmap para Cargos de TI

    Aqui está um roadmap geral para os cargos de TI, incluindo IA, Júnior, Pleno e Sênior:

    ### Cientista de Dados
    - Júnior: Conhecimentos básicos em estatística e linguagens de programação.
    - Pleno: Experiência em análise de dados e machine learning.
    - Sênior: Especialização em áreas específicas de ciência de dados e liderança de projetos.

    ### Desenvolvedor de Software
    - Júnior: Conhecimentos básicos em uma linguagem de programação.
    - Pleno: Experiência no desenvolvimento de aplicativos web ou móveis.
    - Sênior: Especialização em arquitetura de software e liderança técnica.

    ### Engenheiro de Dados
    - Júnior: Experiência em bancos de dados relacionais e linguagens de consulta.
    - Pleno: Conhecimento em ferramentas de big data e processamento distribuído.
    - Sênior: Especialização em design e otimização de pipelines de dados.

    ### Analista de Segurança da Informação
    - Júnior: Conhecimento básico em segurança de redes e sistemas.
    - Pleno: Experiência em análise de vulnerabilidades e resposta a incidentes.
    - Sênior: Especialização em arquitetura de segurança e gestão de riscos.

    ### Administrador de Redes
    - Júnior: Conhecimentos básicos em configuração de redes e dispositivos.
    - Pleno: Experiência em administração de servidores e gerenciamento de redes.
    - Sênior: Especialização em design e implementação de infraestrutura de rede.

    """)

#TODO Page - initial
# ------------------------------------------------------------------------------
# Function for the initial page
def initial_page():
    st.title("Bem-vindo ao Projeto LTD!")
    
    # Descrição da Estácio
    st.markdown("""
    ## Sobre a Estácio

    A Estácio é uma instituição de ensino superior comprometida em oferecer educação de qualidade e acessível para todos. Com uma ampla gama de cursos e programas, a Estácio prepara os alunos para enfrentar os desafios do mercado de trabalho e alcançar seus objetivos profissionais.
    """)
    
    # Descrição do Projeto LTD
    st.markdown("""
    ## Sobre o Projeto LTD

    O Projeto LTD é uma iniciativa da Estácio que visa combinar tecnologia e educação para fornecer soluções inovadoras aos alunos e à comunidade em geral. Este semestre, o foco do projeto é desenvolver ferramentas de inteligência artificial (IA) para auxiliar na recolocação e no desenvolvimento profissional de membros da comunidade e profissionais em busca de aprimoramento.
    """)
    
    # Adicionar seção de funcionalidades do projeto
    st.header("Funcionalidades do Projeto")
    st.markdown("""
    O Projeto LTD deste semestre apresenta duas principais funcionalidades:

    ### Chatbot com Inteligência Artificial

    O chatbot com inteligência artificial foi projetado para oferecer assistência personalizada aos usuários, fornecendo informações sobre oportunidades de emprego, dicas de carreira, cursos disponíveis e muito mais. Ele é capaz de responder a uma variedade de perguntas e fornecer orientações relevantes para ajudar os usuários em suas jornadas profissionais.

    ### Gerador de Currículo em PDF

    O gerador de currículo em PDF é uma ferramenta prática para criar currículos profissionais de forma rápida e fácil. Os usuários podem preencher informações sobre sua experiência profissional, habilidades, educação e outras qualificações relevantes, e o gerador produzirá um currículo formatado profissionalmente em formato PDF pronto para ser enviado para potenciais empregadores.
    """)
    
    # Adicionar imagens dos LTDs passados
    st.header("LTDs do Passado")
    st.markdown("""
    Aqui estão algumas imagens de LTDs do passado:
    """)
    
    ltd_images = [
        "https://example.com/ltd1.jpg",
        "https://example.com/ltd2.jpg",
        "https://example.com/ltd3.jpg"
    ]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(ltd_images[0], use_column_width=True)
    with col2:
        st.image(ltd_images[1], use_column_width=True)
    with col3:
        st.image(ltd_images[2], use_column_width=True)
    
#TODO Page - WhatsApp AI
# ------------------------------------------------------------------------------
# Function for the WhatsApp AI Bot Help page
def whatsapp_ai_bot_help_page():
    page = st.sidebar.radio("Selecione uma página", ["Chatbot com Interface Whatsapp Web", "Chatbot com Interface IA", "Sobre a Automação com WhatsApp", "Utilizando IA para Respostas", "Usando o Typebot"])

    if page == "Chatbot com Interface IA":
        chatbotGemeni()

    if page == "Sobre a Automação com WhatsApp":
        st.title("Sobre a Automação com WhatsApp")
        st.markdown("""
        A automação com WhatsApp permite automatizar interações e respostas no WhatsApp. Aqui está como começar:

        1. Integre uma plataforma de automação, como Twilio ou ChatGPT, com o WhatsApp Business API.
        2. Configure respostas automáticas para mensagens recebidas com base em palavras-chave ou padrões.
        3. Implemente fluxos de conversação para guiar os usuários através de interações automatizadas.

        Com a automação do WhatsApp, você pode melhorar a eficiência e a experiência do usuário em suas interações comerciais.

        ---""")
    
    elif page == "Utilizando IA para Respostas":
            st.title("Utilizando IA para Respostas")
            st.markdown("""
            A inteligência artificial (IA) pode ser integrada ao WhatsApp para fornecer respostas automáticas avançadas. Aqui está como fazer isso:
    
            1. Treine um modelo de IA com dados de perguntas frequentes e suas respostas correspondentes.
            2. Implemente o modelo treinado em uma plataforma de automação, como Twilio ou Dialogflow.
            3. Configure gatilhos para acionar respostas do modelo de IA com base nas mensagens recebidas.
    
            Com a IA, é possível oferecer respostas mais sofisticadas e personalizadas aos usuários do WhatsApp.
    
            ---""")
            st.markdown("""
            Neste projeto, optei por utilizar a API do Gemini do Google para implementar a inteligência artificial. A API do Gemini oferece funcionalidades básicas gratuitas, como tradução de texto e detecção de idioma, o que a torna uma opção adequada para este propósito específico.
    
            Para obter acesso mais amplo a recursos avançados de IA, como geração de texto avançada, a API do ChatGPT pode ser uma escolha ideal, embora tenha um modelo de precificação baseado no uso, incluindo uma opção gratuita limitada e planos pagos para acesso mais amplo.
            """)

    elif page == "Usando o Typebot":
        st.title("Usando o Typebot")
        st.markdown("""
        O Typebot é uma plataforma de criação de chatbots que pode ser integrada ao WhatsApp. Aqui está como começar:

        1. Crie um chatbot personalizado no Typebot com respostas automáticas para perguntas frequentes.
        2. Integre o chatbot do Typebot com o WhatsApp Business API usando as ferramentas de integração fornecidas.
        3. Configure as regras de encaminhamento para direcionar mensagens recebidas no WhatsApp para o chatbot do Typebot.

        Com o Typebot, é possível criar e gerenciar chatbots poderosos para interações automatizadas no WhatsApp.

        ---""")

    elif page == "Chatbot com Interface Whatsapp Web":
        st.title("Chatbot com Interface Web do WhatsApp")
        st.markdown("""
        A interface web do WhatsApp permite interagir com um chatbot diretamente no navegador. Aqui está como acessar o chatbot:

        1. Abra o seguinte link em seu navegador: [Chatbot com Interface Web do WhatsApp](https://typebot.co/whatsapp-ltd-estacio-ia)
        2. Use o seu smartphone para escanear o QrCode abaixo e acessar o chatbot.
        
        """)
        st.image("qrcode_typebot.co.png", width=300)

#TODO Page - Dashboard
# ----------------------------------------------------------------------------------------------------------------------
def dash():
    pass

#TODO Page - Study Material
# ------------------------------------------------------------------------------
# Function for the Study Material page
def study_material_page():
    st.title("Material de Estudos em TI")

    st.markdown("""

    Aqui está uma lista de recursos de estudo na área de Tecnologia da Informação, organizados por nível:

    ## Iniciante

    ### Vídeo Aulas
    - [Curso de Desenvolvimento Web - TreinaWeb](https://www.treinaweb.com.br/curso/desenvolvimento-web)
    - [Curso de Python - Udemy](https://www.udemy.com/course/python-para-todos/)
    - [Curso de HTML5 e CSS3 - Origamid](https://www.origamid.com/curso/html5-css3)
    
    ### Livros
    - "Python Fluente" - Luciano Ramalho
    - "HTML5 e CSS3: Domine a web do futuro" - Ricardo R. Lecheta

    ### Cursos Online
    - [Python Fundamentos - Udemy](https://www.udemy.com/course/python-fundamentos/)
    - [HTML5 e CSS3 - Udacity](https://br.udacity.com/course/intro-to-html-and-css--ud001)

    ## Intermediário

    ### Vídeo Aulas
    - [Curso de React - Rocketseat](https://rocketseat.com.br/starter)
    - [Curso de JavaScript - Danki Code](https://www.dankicode.com/curso-completo-de-javascript)
    - [Curso de Data Science - Data Science Academy](https://www.datascienceacademy.com.br/course?courseid=python-fundamentos)

    ### Livros
    - "Clean Code: A Handbook of Agile Software Craftsmanship" - Robert C. Martin
    - "JavaScript: The Good Parts" - Douglas Crockford

    ### Cursos Online
    - [React Native - Udemy](https://www.udemy.com/course/react-native-app/)
    - [Data Science e Machine Learning - Coursera](https://www.coursera.org/learn/machine-learning)

    ## Avançado

    ### Vídeo Aulas
    - [Curso de Inteligência Artificial - Udacity](https://br.udacity.com/course/intro-to-artificial-intelligence--cs271)
    - [Curso de Docker - Alura](https://www.alura.com.br/curso-online-docker-e-docker-compose)
    - [Curso de Deep Learning - Data Science Academy](https://www.datascienceacademy.com.br/course?courseid=deep-learning-ii)

    ### Livros
    - "Deep Learning" - Ian Goodfellow, Yoshua Bengio, Aaron Courville
    - "Docker in Action" - Jeff Nickoloff

    ### Cursos Online
    - [Machine Learning - Coursera](https://www.coursera.org/learn/machine-learning)
    - [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)

    """)

#TODO Page - Technology News
# ------------------------------------------------------------------------------
# Function for the Technology News page
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

#TODO Page - Hacker Prevention
# ------------------------------------------------------------------------------
# Function to set theme to green and black
def set_hacker_theme():
    # Define custom CSS styles
    hacker_custom_css = f"""
        body {{
            background-color: #000000;
            color: #00FF00;
        }}
        .streamlit-button {{
            color: #00FF00;
            border-color: #00FF00;
        }}
        .streamlit-button:hover {{
            color: #00FF00;
            background-color: #000000;
        }}
        .streamlit-button:active {{
            color: #00FF00;
            background-color: #000000;
        }}
    """

    # Set custom CSS styles
    st.markdown(f'<style>{hacker_custom_css}</style>', unsafe_allow_html=True)

# Function for the Hacker Prevention page
def hacker_prevention_page():
    # Set theme to green and black
    set_hacker_theme()

    st.title("Prevenção Contra Ataques de Hacker e Segurança na Dark Web e Deep Web")

    # Sidebar navigation
    st.sidebar.title('Navegação')
    page = st.sidebar.radio('Ir para:', ('Prevenção de Ataques de Hacker', 'Dark Web e Deep Web'))

    if page == 'Prevenção de Ataques de Hacker':
        st.video("https://www.youtube.com/watch?v=7V4jWIYhX9c")

        st.markdown("""
        ## Prevenção Contra Ataques de Hacker

        Aqui estão algumas dicas para proteger seus sistemas e dados contra ataques de hackers:

        ### 1. Mantenha Seu Software Atualizado
        Mantenha todos os softwares, incluindo sistemas operacionais, navegadores da web e aplicativos, atualizados com as últimas atualizações de segurança. As atualizações frequentes ajudam a corrigir vulnerabilidades conhecidas.

        ### 2. Use Senhas Fortes
        Use senhas fortes e únicas para todas as suas contas online. Evite usar senhas óbvias ou fáceis de adivinhar, e considere usar um gerenciador de senhas para armazenar senhas com segurança.

        ### 3. Tome Cuidado com Phishing
        Esteja atento a e-mails de phishing e mensagens suspeitas que solicitam informações pessoais ou credenciais de login. Nunca clique em links suspeitos ou baixe anexos de fontes não confiáveis.
        """)

    elif page == 'Dark Web e Deep Web':
        st.markdown("""
        ## Dark Web e Deep Web

        ### O Que É a Dark Web?
        A Dark Web é uma parte da internet que não é acessível por meio de motores de busca convencionais, como o Google. É conhecida por ser um ambiente onde atividades ilegais, como venda de drogas, armas e informações roubadas, podem ocorrer.

        ### O Que É a Deep Web?
        A Deep Web é uma parte da internet que não é indexada pelos motores de busca tradicionais. Isso inclui sites protegidos por senhas, bancos de dados privados e conteúdo não acessível ao público em geral.

        ### Como Se Prevenir na Dark Web e Deep Web?
        - Evite acessar a Dark Web, pois ela pode expor você a atividades ilegais e conteúdo perigoso.
        - Nunca compartilhe informações pessoais ou confidenciais em sites da Dark Web ou Deep Web.
        - Mantenha seus dispositivos protegidos com software antivírus e firewall atualizados.
        - Evite clicar em links suspeitos e baixar arquivos de fontes não confiáveis ao navegar na internet.
        - Considere usar uma VPN (rede virtual privada) para proteger sua privacidade ao navegar online.

        ### Como Acessar a Dark Web e Deep Web de Forma Segura?
        Se você deseja acessar a Dark Web ou Deep Web por razões legítimas, siga estas precauções:
        - Use um navegador especializado, como o Tor Browser, que oferece anonimato e criptografia.
        - Nunca forneça informações pessoais ou financeiras ao acessar sites na Dark Web ou Deep Web.
        - Evite clicar em links desconhecidos e verifique a reputação dos sites antes de acessá-los.
        """)

# Main function
def main():
    st.sidebar.image("estacio.jpg", use_column_width=True)
    st.sidebar.title("Menu")
    st.sidebar.markdown("""
        ## Sobre o Projeto LTD
        
        O Projeto Laboratório de Transformação Digital (LTD) é uma iniciativa da Estácio que visa integrar tecnologia e educação para oferecer soluções inovadoras aos alunos e à comunidade. Este semestre, o foco do projeto é desenvolver ferramentas de inteligência artificial para auxiliar na recolocação e no desenvolvimento profissional.
    
        ## Funcionalidades do Projeto
        
        - **Chatbot com Inteligência Artificial:** Oferece assistência personalizada aos usuários, fornecendo informações sobre oportunidades de emprego, dicas de carreira e cursos disponíveis.
        - **Gerador de Currículo em PDF:** Permite criar currículos profissionais de forma rápida e fácil, facilitando a busca por emprego.
    """)
    
    selected_page = st.sidebar.radio("Selecione uma página", ["🏠 Início",
        "💼 Jobs",
        "📚 Material Estudos", 
        "💻 Notícias",
        "🔗 Sobre",
        "🛠️ Ferramentas de IA",
        "📱 Dashboard",
        "📄 Gerador de Currículo",
        "🤖 ChatBot",
        "👿 Darknet"], index=0)
    if selected_page == "🏠 Início":
        initial_page()
    elif selected_page == "📄 Gerador de Currículo":
        curriculoVintage.curriculo()
    elif selected_page == "🔗 Sobre":
        about_page()
    elif selected_page == "📱 Dashboard":
        dash()
    elif selected_page == "💼 Jobs":
        it_jobs_page()
    elif selected_page == "📚 Material Estudos":
        study_material_page()
    elif selected_page == "💻 Notícias":
        technology_news_page()
    elif selected_page == "🤖 ChatBot":
        whatsapp_ai_bot_help_page()
    elif selected_page == "👿 Darknet":
        hacker_prevention_page()
    else:
        ai_tools_page()

if __name__ == "__main__":
    main()
