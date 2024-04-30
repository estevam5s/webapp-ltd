# Import required libraries
from dotenv import load_dotenv
from io import BytesIO
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from reportlab.lib.pagesizes import letter
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
def init_chatbot_model():
    return ChatOpenAI(
        temperature=0.5,
        model_name="gpt-3.5-turbo"
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

# Function for the chatbot page
def chatbot_page():
    st.title("ChatBot LTD")

    init_chatbot_session_state()

    chat = init_chatbot_model()

    user_input = st.text_input('YOU: ', key='prompt_input')

    if st.button("Enviar"):
        st.session_state.entered_prompt = user_input

    if st.session_state.entered_prompt != "":
        user_query = st.session_state.entered_prompt
        st.session_state.past.append(user_query)
        output = generate_response(chat)
        st.session_state.generated.append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i],
                    is_user=True, key=str(i) + '_user')

    st.markdown("""
    ---
    Feito por [Estevam Souza](https://github.com/estevam5s)""")

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

# Function for the initial page
def initial_page():
    st.title("Bem-vindo ao Projeto LTD!")
    st.write("Este é um projeto que combina um chatbot com um gerador de currículo em PDF. Utilize o menu à esquerda para navegar pelas diferentes funcionalidades do projeto.")

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
    selected_page = st.sidebar.radio("Selecione uma página", [("Início 🏠", "Início"), ("ChatBot 💬", "ChatBot"), ("Gerador de Currículo 📄", "Gerador de Currículo"), ("Sobre ℹ️", "Sobre"), ("Ferramentas de IA 🛠️", "Ferramentas de IA"), ("Dashboard 📱", "Dashboard")], index=0)
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
    else:
        ai_tools_page()

if __name__ == "__main__":
    main()
