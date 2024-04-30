from setuptools import setup, find_packages

# Leitura do README.md para incluir como descrição longa do projeto
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='meu-projeto',
    version='0.1',
    author='Seu Nome',
    author_email='seu@email.com',
    description='Uma breve descrição do seu projeto',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/seuusuario/meu-projeto',
    packages=find_packages(),
    install_requires=[
        'reportlab',
        # outras dependências do seu projeto
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
