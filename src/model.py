import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib 
import os
import numpy as np


# Carregando os dados
df = pd.read_csv('../data/dados_transformados.csv')

# Seleção das variáveis independentes e dependentes
X = df[['latitude', 'longitude', 'minimo_noites', 'reviews_por_mes',
        'numero_de_reviews', 'disponibilidade_365', 'room_type',
        'bairro', 'bairro_group']]
y = df['price']

# Codificação de variáveis categóricas (transformando 'room_type', 'bairro' e 'bairro_group' em variáveis binárias)
X = pd.get_dummies(X)

# Divisão dos dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento do modelo
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Previsões e avaliação
y_pred = model.predict(X_test_scaled)

# Cálculo das métricas de erro
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE

print("Erro absoluto médio (MAE):", mae)
print("Erro quadrático médio (RMSE):", rmse)

# Definir o caminho para os arquivos de modelo e scaler
model_dir = '../models/'

# Verificar se o diretório existe, caso contrário, criar
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Salvar o modelo e o scaler com a extensão .pkl
joblib.dump(model, os.path.join(model_dir, 'modelo.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))