import streamlit as st

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