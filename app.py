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


# Load environment variables
load_dotenv()
st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

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
    Feito com 🤖 por [Estevam Souza](https://github.com/estevam5s)""")

# Function for the resume generator page
def generate_resume_page():
    st.title("Gerador de Currículo")

    init_resume_session_state()

    name = st.text_input("Nome completo:", value=st.session_state['name'])
    email = st.text_input("E-mail:", value=st.session_state['email'])
    phone = st.text_input("Telefone:", value=st.session_state['phone'])
    experience = st.text_area("Experiência Profissional:", value=st.session_state['experience'])
    education = st.text_area("Educação:", value=st.session_state['education'])
    skills = st.text_area("Habilidades:", value=st.session_state['skills'])

    if st.button("Gerar Currículo em PDF"):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)

        data = [
            ["Nome:", name],
            ["E-mail:", email],
            ["Telefone:", phone],
            ["Experiência Profissional:", experience],
            ["Educação:", education],
            ["Habilidades:", skills]
        ]

        table = Table(data, colWidths=150, rowHeights=30)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ]))

        doc.build([table])

        buffer.seek(0)
        st.download_button(label="Baixar Currículo em PDF", data=buffer, file_name="curriculo.pdf", mime="application/pdf", key=None)

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
    Feito com ❤️ por [Seu Nome](https://github.com/seuusuario)""")

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

# Function for the initial page
def initial_page():
    st.title("Bem-vindo ao Projeto LTD!")
    
    # Adicionando informações sobre o Projeto LTD
    st.markdown("""
    ## Sobre o Projeto LTD

    O Projeto LTD é uma iniciativa da Estácio que visa combinar tecnologia e educação para fornecer soluções inovadoras aos alunos e à comunidade em geral. O projeto inclui um chatbot com inteligência artificial para assistência personalizada e um gerador de currículo em PDF para ajudar os usuários a criar currículos profissionais de forma rápida e fácil.

    ## Sobre a Estácio

    A Estácio é uma instituição de ensino superior comprometida em oferecer educação de qualidade e acessível para todos. Com uma ampla gama de cursos e programas, a Estácio prepara os alunos para enfrentar os desafios do mercado de trabalho e alcançar seus objetivos profissionais.

    ## LTDs do Passado

    Aqui estão algumas imagens de LTDs do passado:

    ![LTD 1](https://example.com/ltd1.jpg)
    ![LTD 2](https://example.com/ltd2.jpg)
    ![LTD 3](https://example.com/ltd3.jpg)

    """)

# Function for the WhatsApp AI Bot Help page
def whatsapp_ai_bot_help_page():
    st.title("Ajuda do WhatsApp AI Bot")

    st.markdown("""
    # Ajuda do WhatsApp AI Bot

    Você pode usar o bot de WhatsApp AI para obter respostas para suas perguntas. Aqui está como:

    1. Adicione o número de WhatsApp do bot à sua lista de contatos.
    2. Envie uma mensagem para o bot com sua pergunta.
    3. O bot responderá automaticamente com uma resposta baseada na inteligência artificial.

    Certifique-se de incluir informações claras e concisas em suas mensagens para obter as melhores respostas do bot.

    ## Links Úteis
    - [Adicionar Bot do WhatsApp](https://api.whatsapp.com/send/?phone=seunumerodewhatsapp)
    - [FAQ do Projeto LTD da Estácio](#faq)

    ---""")

    st.markdown("""
    ## FAQ do Projeto LTD da Estácio

    ### 1. Qual é o objetivo do projeto LTD?

    O objetivo do projeto LTD é fornecer uma plataforma que combina um chatbot com inteligência artificial e um gerador de currículo em PDF para ajudar os usuários com suas necessidades de informações e criação de currículos.

    ### 2. Quem está por trás do projeto LTD?

    O projeto LTD é desenvolvido por uma equipe da Estácio, liderada pelo desenvolvedor Estevam Souza.

    ### 3. O bot de WhatsApp AI responde a todas as perguntas?

    O bot de WhatsApp AI foi treinado para responder a uma variedade de perguntas, mas pode não ter resposta para todas as consultas. Certifique-se de incluir informações claras em suas mensagens para obter as melhores respostas.

    ---""")

# Dashboard
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

# ----------------------------------------------------------------------------------------------------------------------

# Main function
def main():
    st.sidebar.image("estacio.jpg", use_column_width=True)
    st.sidebar.title("Menu")
    st.sidebar.markdown("""
    - Implementação da página inicial do projeto.
    - Adição de informações sobre o projeto LTD.
    - Adicionar descrição e foto na barra lateral.
    - Incluir opções de navegação para outras páginas.
    """)
    selected_page = st.sidebar.radio("Selecione uma página", [("Início 🏠", "Início"), ("Jobs 💼", "Jobs"), ("ChatBot 💬", "ChatBot"), ("Whatsapp 💬", "Whatsapp"), ("Gerador de Currículo 📄", "Gerador de Currículo"), ("Sobre ℹ️", "Sobre"), ("Ferramentas de IA 🛠️", "Ferramentas de IA"), ("Dashboard 📱", "Dashboard")], index=0)
    if selected_page[1] == "Início":
        initial_page()
    elif selected_page[1] == "ChatBot":
        chatbot_page()
    elif selected_page[1] == "Gerador de Currículo":
        generate_resume_page()
    elif selected_page[1] == "Sobre":
        about_page()
    elif selected_page[1] == "Dashboard":
        dash()
    elif selected_page[1] == "Jobs":
        it_jobs_page()
    elif selected_page[1] == "Whatsapp":
        whatsapp_ai_bot_help_page()
    else:
        ai_tools_page()

if __name__ == "__main__":
    main()
