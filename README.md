# Light-House Project
Este projeto tem como objetivo a previsão do preço de imóveis da cidade de Nova York com base em diversas características, utilizando técnicas de análise exploratória de dados (EDA), modelagem e avaliação de modelos de machine learning.


## Estrutura do Repositório
Light-House-Project/
├── README.md                      # Este arquivo de documentação
├── requirements.txt               # Arquivo de dependências do projeto
├── notebooks/                     
│   ├── analise-exploratoria.ipynb          # Análises exploratórias e estatísticas
├── src/
│   ├── model.py                   # Código de treinamento e avaliação do modelo
│   └── teste.py                   # Código para testar previsões (com dados CSV e JSON)
├── models/
│   ├── modelo.pkl                 # Modelo treinado salvo
│   └── scaler.pkl                 # Scaler para normalização dos dados
└── data/
    ├── dados_transformados.csv    # Dados transformados para treinamento e teste
    └── teste_indicium_precificacao.csv  # Arquivo base fornecido


## Instalação
Para rodar este projeto, siga os passos abaixo:
1. Clonar o repositório:
    bash
    git clone https://github.com/Ewaldouhlmann/LH_CD_EWALDOUHLMANN
    cd LH_CD_EWALDOUHLMANN
2. Criar um ambiente virtual
    python -m venv venv
    Ativar o ambiente virtual
    source venv/bin/activate (Linux/Mac) ou venv\Scripts\activate (Windows)

3. Instalar as dependências
    pip install -r requirements.txt

4. Rodar o projeto (notebook ou código):
    jupyter notebook (para abrir o notebook)
    python src/model.py (para treinar o modelo)
    python src/teste.py (para testar o modelo com dados CSV e JSON)

