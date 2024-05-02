# Import required libraries
from dotenv import load_dotenv
from io import BytesIO
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
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
    st.title("Sobre")
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
def ai_tools_page():
    st.title("Ferramentas de IA")
    st.markdown("""
    # Ferramentas de IA

    Aqui estão algumas ferramentas de inteligência artificial que podem ser úteis:

    - [OpenAI](https://openai.com): Plataforma de inteligência artificial que oferece uma série de modelos e ferramentas para desenvolvedores.
    - [TensorFlow](https://www.tensorflow.org): Uma biblioteca de software de código aberto para aprendizado de máquina e inteligência artificial desenvolvida pelo Google.
    - [PyTorch](https://pytorch.org): Uma biblioteca de aprendizado de máquina de código aberto baseada na linguagem de programação Python.
    - [Scikit-learn](https://scikit-learn.org): Uma biblioteca de aprendizado de máquina de código aberto para a linguagem de programação Python.
    - [NLTK](https://www.nltk.org): Uma plataforma líder para construção de programas Python para trabalhar com dados de linguagem humana.

    Estas são apenas algumas das muitas ferramentas disponíveis. Certifique-se de explorar mais para encontrar as que melhor se adequam às suas necessidades.

    ---""")

#TODO Page - IT Jobs
# ------------------------------------------------------------------------------
# Function for the IT Jobs page
def it_jobs_page():
    st.title("Vagas de Emprego em TI")

    st.markdown("""
    ## Vagas de Emprego em TI

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
    page = st.sidebar.radio("Selecione uma página", ["Sobre a Automação com WhatsApp", "Utilizando IA para Respostas", "Usando o Typebot"])

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

    elif page == "Usando o Typebot":
        st.markdown("""
        # Usando o Typebot

        O Typebot é uma plataforma de criação de chatbots que pode ser integrada ao WhatsApp. Aqui está como começar:

        1. Crie um chatbot personalizado no Typebot com respostas automáticas para perguntas frequentes.
        2. Integre o chatbot do Typebot com o WhatsApp Business API usando as ferramentas de integração fornecidas.
        3. Configure as regras de encaminhamento para direcionar mensagens recebidas no WhatsApp para o chatbot do Typebot.

        Com o Typebot, é possível criar e gerenciar chatbots poderosos para interações automatizadas no WhatsApp.

        ---""")


#TODO Page - Dashboard
# ----------------------------------------------------------------------------------------------------------------------
def dash():

    # ---- READ EXCEL ----
    @st.cache_data
    def get_data_from_excel():
        df = pd.read_excel(
            io="supermarkt_sales.xlsx",
            engine="openpyxl",
            sheet_name="Sales",
            skiprows=3,
            usecols="B:R",
            nrows=1000,
        )
        # Add 'hour' column to dataframe
        df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
        return df

    df = get_data_from_excel()

    # ---- SIDEBAR ----
    st.sidebar.header("Please Filter Here:")
    city = st.sidebar.multiselect(
        "Select the City:",
        options=df["City"].unique(),
        default=df["City"].unique()
    )

    customer_type = st.sidebar.multiselect(
        "Select the Customer Type:",
        options=df["Customer_type"].unique(),
        default=df["Customer_type"].unique(),
    )

    gender = st.sidebar.multiselect(
        "Select the Gender:",
        options=df["Gender"].unique(),
        default=df["Gender"].unique()
    )

    df_selection = df.query(
        "City == @city & Customer_type ==@customer_type & Gender == @gender"
    )

    # Check if the dataframe is empty:
    if df_selection.empty:
        st.warning("No data available based on the current filter settings!")
        st.stop() # This will halt the app from further execution.

    # ---- MAINPAGE ----
    st.title(":bar_chart: Sales Dashboard")
    st.markdown("##")

    # TOP KPI's
    total_sales = int(df_selection["Total"].sum())
    average_rating = round(df_selection["Rating"].mean(), 1)
    star_rating = ":star:" * int(round(average_rating, 0))
    average_sale_by_transaction = round(df_selection["Total"].mean(), 2)

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Total Sales:")
        st.subheader(f"US $ {total_sales:,}")
    with middle_column:
        st.subheader("Average Rating:")
        st.subheader(f"{average_rating} {star_rating}")
    with right_column:
        st.subheader("Average Sales Per Transaction:")
        st.subheader(f"US $ {average_sale_by_transaction}")

    st.markdown("""---""")

    # SALES BY PRODUCT LINE [BAR CHART]
    sales_by_product_line = df_selection.groupby(by=["Product line"])[["Total"]].sum().sort_values(by="Total")
    fig_product_sales = px.bar(
        sales_by_product_line,
        x="Total",
        y=sales_by_product_line.index,
        orientation="h",
        title="<b>Sales by Product Line</b>",
        color_discrete_sequence=["#0083B8"] * len(sales_by_product_line),
        template="plotly_white",
    )
    fig_product_sales.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # SALES BY HOUR [BAR CHART]
    sales_by_hour = df_selection.groupby(by=["hour"])[["Total"]].sum()
    fig_hourly_sales = px.bar(
        sales_by_hour,
        x=sales_by_hour.index,
        y="Total",
        title="<b>Sales by hour</b>",
        color_discrete_sequence=["#0083B8"] * len(sales_by_hour),
        template="plotly_white",
    )
    fig_hourly_sales.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),
    )


    left_column, right_column = st.columns(2)
    left_column.plotly_chart(fig_hourly_sales, use_container_width=True)
    right_column.plotly_chart(fig_product_sales, use_container_width=True)

    # ---- HIDE STREAMLIT STYLE ----
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

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
    st.title("Notícias sobre Tecnologia")

    # Sidebar navigation
    st.sidebar.title('Navegação')
    page = st.sidebar.radio('Ir para:', ('Últimas Notícias', 'Detalhes'))

    if page == 'Últimas Notícias':
        st.markdown("""
        ## Notícias sobre Tecnologia

        Aqui estão algumas das últimas notícias sobre tecnologia:

        ### 1. Novo iPhone 14 Anunciado pela Apple
        A Apple anunciou o lançamento do novo iPhone 14, que promete recursos avançados e melhorias significativas em relação aos modelos anteriores. O iPhone 14 apresenta uma nova tela OLED de alta resolução e uma câmera aprimorada com capacidades de fotografia computacional.

        ### 2. Google Revela Avanços em IA
        O Google revelou avanços impressionantes em inteligência artificial, incluindo um novo algoritmo de aprendizado de máquina capaz de superar desafios complexos de jogos de tabuleiro. Os pesquisadores do Google afirmam que o novo algoritmo demonstra uma capacidade sem precedentes de aprendizado e adaptação.

        ### 3. Amazon Lança Novo Dispositivo de Casa Inteligente
        A Amazon lançou um novo dispositivo de casa inteligente chamado Echo Hub, projetado para ser o centro de controle para dispositivos domésticos conectados. O Echo Hub oferece recursos avançados de voz e integração perfeita com outros dispositivos compatíveis com Alexa.

        ### 4. Microsoft Anuncia Parceria com Empresa de Robótica
        A Microsoft anunciou uma parceria estratégica com uma empresa líder em robótica para desenvolver soluções inovadoras para automação industrial e logística. A parceria visa combinar a expertise em software da Microsoft com a experiência em hardware da empresa de robótica para criar soluções de ponta.

        ### 5. Facebook Lança Novo Recurso de Realidade Aumentada
        O Facebook lançou um novo recurso de realidade aumentada chamado AR Studio, que permite aos usuários criar e compartilhar experiências imersivas de RA diretamente do aplicativo. O AR Studio oferece uma ampla gama de ferramentas e recursos para criar experiências interativas e envolventes.
        """)

    elif page == 'Detalhes':
        st.markdown("""
        ## Detalhes

        Aqui estão alguns detalhes adicionais sobre as notícias:

        ### Novo iPhone 14
        A Apple anunciou várias novas características interessantes para o iPhone 14, incluindo uma nova tela OLED de alta resolução e uma câmera aprimorada com capacidades de fotografia computacional.

        ### Avanços em IA
        Os avanços em inteligência artificial revelados pelo Google têm o potencial de impactar uma ampla gama de indústrias, desde jogos até medicina e logística.

        ### Novo Dispositivo de Casa Inteligente da Amazon
        O Echo Hub da Amazon promete tornar a automação residencial mais acessível e conveniente para os usuários, oferecendo uma maneira fácil de controlar dispositivos domésticos inteligentes por meio de comandos de voz.

        ### Parceria da Microsoft com Empresa de Robótica
        A parceria estratégica entre a Microsoft e uma empresa de robótica sugere uma crescente ênfase na automação e na integração entre software e hardware para impulsionar a eficiência industrial.

        ### Recurso de Realidade Aumentada do Facebook
        O AR Studio do Facebook tem o potencial de transformar a forma como as pessoas interagem com a mídia social, permitindo a criação e o compartilhamento de experiências imersivas de realidade aumentada diretamente do aplicativo.
        """)


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
        "💬 Whatsapp",
        "🤖 ChatBot",
        "👿 Darknet"], index=0)
    if selected_page == "🏠 Início":
        initial_page()
    elif selected_page == "💬 ChatBot":
        chatbotGemeni()
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
    elif selected_page == "💬 Whatsapp":
        whatsapp_ai_bot_help_page()
    elif selected_page == "👿 Darknet":
        hacker_prevention_page()
    else:
        ai_tools_page()

if __name__ == "__main__":
    main()
