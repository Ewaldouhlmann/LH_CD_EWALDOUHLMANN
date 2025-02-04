import pandas as pd
import joblib
import os

def carregar_modelo_e_scaler(modelo_path, scaler_path):
    """
    Carrega o modelo e o scaler salvos em disco.

    Parâmetros:
      - modelo_path (str): Caminho para o modelo salvo.
      - scaler_path (str): Caminho para o scaler salvo.

    Retorno:
      - model: Modelo carregado.
      - scaler: Scaler carregado.
    """
    try:
        model = joblib.load(modelo_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f'Erro ao carregar o modelo e o scaler: {e}')
        return None, None

def preparar_dados(dados_imovel, scaler):
    """
    Prepara os dados do imóvel para serem utilizados em um modelo de machine learning.

    Parâmetros:
      - dados_imovel (DataFrame): Dados do imóvel.
      - scaler (Scaler): Scaler utilizado para normalizar os dados.

    Retorno:
      - dados_imovel_scaled (ndarray): Dados do imóvel normalizados.
    """
    try:
        # Codifica as variáveis categóricas (sem omitir a primeira categoria)
        dados_imovel = pd.get_dummies(dados_imovel, drop_first=False)

        # Obter as colunas usadas no scaler
        colunas_necessarias = scaler.feature_names_in_

        # Identificar as colunas faltantes e preenchê-las com 0 de uma vez
        missing_cols = [col for col in colunas_necessarias if col not in dados_imovel.columns]
        if missing_cols:
            df_missing = pd.DataFrame(0, index=dados_imovel.index, columns=missing_cols)
            dados_imovel = pd.concat([dados_imovel, df_missing], axis=1)

        # Organiza as colunas na mesma ordem que o scaler espera
        dados_imovel = dados_imovel[colunas_necessarias]

        # Desfragmenta o DataFrame criando uma cópia
        dados_imovel = dados_imovel.copy()

        # Normaliza os dados
        dados_imovel_scaled = scaler.transform(dados_imovel)

        return dados_imovel_scaled
    except Exception as e:
        print(f'Erro ao preparar os dados: {e}')
        return None

def prever_preco(dados_imovel, model, scaler):
    """
    Faz a previsão do preço usando os dados do imóvel, o modelo e o scaler.

    Retorno:
      - previsao: Valor previsto.
    """
    try:
        dados_imovel_scaled = preparar_dados(dados_imovel, scaler)
        previsao = model.predict(dados_imovel_scaled)
        return previsao
    except Exception as e:
        print(f'Erro ao fazer a previsão: {e}')
        return None

def testar_json(json_data, model, scaler):
    """
    Converte o dicionário JSON (Python dict) em DataFrame, seleciona as colunas necessárias,
    faz a previsão e retorna o resultado.

    Parâmetros:
      - json_data (dict): Dados do imóvel.
      - model: Modelo carregado.
      - scaler: Scaler carregado.

    Retorno:
      - previsao: Valor previsto para o imóvel.
    """
    try:
        # Converte o dicionário em um DataFrame com uma única linha
        df_imovel = pd.DataFrame([json_data])
        
        # Seleciona as colunas relevantes conforme utilizado no treinamento
        colunas_entrada = ['latitude', 'longitude', 'minimo_noites', 'reviews_por_mes',
                            'numero_de_reviews', 'room_type',
                            'bairro', 'bairro_group']
        df_imovel = df_imovel.reindex(columns=colunas_entrada)

        # Faz a previsão e retorna o resultado
        previsao = prever_preco(df_imovel, model, scaler)
        return previsao
    except Exception as e:
        print(f'Erro ao testar o JSON: {e}')

def main():
    # Caminhos para os arquivos
    modelo_path = '../models/modelo.pkl'
    scaler_path = '../models/scaler.pkl'
    dados_path = '../data/dados_transformados.csv'  

    # Carregar o modelo e o scaler
    model, scaler = carregar_modelo_e_scaler(modelo_path, scaler_path)
    if model is None or scaler is None:
        print("Erro ao carregar modelo ou scaler. O programa será encerrado.")
        return

    # Carregar os dados transformados
    dados_imoveis = pd.read_csv(dados_path)

    # Selecionar as 10 primeiras linhas
    dados_imoveis_top10 = dados_imoveis.head(10)
    
    # Extrair os valores reais da coluna 'price' antes de remover a coluna
    valores_reais = dados_imoveis_top10['price'].tolist()

    # Definir as colunas de entrada (excluindo 'price')
    colunas_entrada = ['latitude', 'longitude', 'minimo_noites', 'reviews_por_mes',
                       'numero_de_reviews', 'disponibilidade_365', 'room_type',
                       'bairro', 'bairro_group']
    
    # Selecionar apenas as colunas necessárias para a previsão
    dados_imoveis_top10 = dados_imoveis_top10[colunas_entrada]

    # Fazer a previsão para os 10 primeiros imóveis
    previsao_csv = prever_preco(dados_imoveis_top10, model, scaler)
    for i, p in enumerate(previsao_csv):
        print(f'Imóvel {i + 1} (CSV): Previsão = {p}, Valor Real = {valores_reais[i]}')

    # Exemplo: Previsão para um imóvel usando dados JSON (dicionário Python)
    json_imovel = {
        'id': 2595,
        'nome': 'Skylit Midtown Castle',
        'host_id': 2845,
        'host_name': 'Jennifer',
        'bairro_group': 'Manhattan',
        'bairro': 'Midtown',
        'latitude': 40.75362,
        'longitude': -73.98377,
        'room_type': 'Entire home/apt',
        'minimo_noites': 1,
        'numero_de_reviews': 45,
        'ultima_review': '2019-05-21',
        'reviews_por_mes': 0.38,
        'calculado_host_listings_count': 2,
        'disponibilidade_365': 355
    }
    previsao_json = testar_json(json_imovel, model, scaler)
    if previsao_json is not None:
        print(f'\nPrevisão do preço para o imóvel (JSON): {previsao_json[0]}')

if __name__ == '__main__':
    main()
