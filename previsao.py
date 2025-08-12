import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# --- 1. Carregar Dados ---
# Carrega os conjuntos de dados
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    data_description = open('data_description.txt', 'r').read() # Para referência, não usado diretamente na lógica do script
    sample_submission = pd.read_csv('sample_submission.csv')
    print("Conjuntos de dados carregados com sucesso.")
except FileNotFoundError:
    print("Certifique-se de que 'train.csv', 'test.csv', 'data_description.txt' e 'sample_submission.csv' estão no mesmo diretório.")
    exit()

# Salva os IDs originais do conjunto de teste para submissão
test_ids = test_df['Id']

# Remove a coluna 'Id' de ambos os conjuntos de dados, pois não é uma feature
train_df = train_df.drop('Id', axis=1)
test_df = test_df.drop('Id', axis=1)

# --- 2. Engenharia de Features e Pré-processamento ---

# Combina os dados de treino e teste para um pré-processamento consistente
all_data = pd.concat((train_df.loc[:,'MSSubClass':'SaleCondition'],
                      test_df.loc[:,'MSSubClass':'SaleCondition']))

print(f"Formato dos dados combinados: {all_data.shape}")

# Transforma a variável alvo 'SalePrice' do conjunto de treino usando logaritmo (log1p)
# Isso é crucial porque a métrica de avaliação é o RMSE do log do preço.
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

# Lida com valores ausentes
# Valores numéricos ausentes: Preenche com a mediana
for col in ('LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars',
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    if all_data[col].isnull().any():
        all_data[col] = all_data[col].fillna(all_data[col].median())

# Valores categóricos ausentes: Preenche com 'None' ou 'No' dependendo do contexto
for col in ('Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC',
            'Fence', 'MiscFeature', 'MasVnrType', 'MSZoning', 'Utilities', 'Exterior1st',
            'Exterior2nd', 'KitchenQual', 'Electrical', 'Functional', 'SaleType'):
    if all_data[col].isnull().any():
        all_data[col] = all_data[col].fillna('None')

# Casos específicos para valores ausentes baseados na descrição dos dados
# 'Utilities' tem muito poucos valores diferentes de 'AllPub', a maioria ausente no conjunto de teste. Remove para simplificar.
if 'Utilities' in all_data.columns:
    all_data = all_data.drop('Utilities', axis=1)

# 'OverallQual' e 'OverallCond' já são numéricos. Não há valores ausentes para estes.

# Engenharia de Features: Criando novas features
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                         all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))
all_data['YearBuilt-Remod'] = all_data['YearRemodAdd'] - all_data['YearBuilt']
all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] + all_data['EnclosedPorch'] +
                            all_data['3SsnPorch'] + all_data['ScreenPorch'])
all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# Converte algumas features numéricas em categóricas, pois seus valores representam categorias
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)

# Codificação de rótulos (Label Encoding) para features ordinais onde a ordem importa (baseado na data_description)
# Para simplificar, usaremos one-hot encoding para todas as categóricas abaixo,
# mas para melhor desempenho, a codificação ordinal específica poderia ser aplicada.

# Codifica Features Categóricas usando One-Hot Encoding
all_data = pd.get_dummies(all_data)
print(f"Formato dos dados após one-hot encoding: {all_data.shape}")

# Separa de volta em conjuntos de treino e teste
X_train = all_data[:len(train_df)]
X_test = all_data[len(train_df):]
y_train = train_df['SalePrice']

# Garante que as colunas correspondam entre os conjuntos de treino e teste
# Isso lida com casos onde algumas categorias podem estar presentes no treino mas não no teste, ou vice-versa
train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0

X_test = X_test[train_cols] # Alinha as colunas


# --- 3. Treinamento do Modelo ---

# Regressor Random Forest
print("\nTreinando Regressor Random Forest...")
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1, max_features=0.75, min_samples_leaf=1)
rf_model.fit(X_train, y_train)
print("Treinamento do Random Forest completo.")

# Regressor XGBoost
print("Treinando Regressor XGBoost...")
xgbr_model = xgb.XGBRegressor(objective='reg:square_error', # Alterado de 'reg:squarederror' para 'reg:square_error' (versões mais recentes do XGBoost)
                              n_estimators=2000,
                              learning_rate=0.01,
                              max_depth=5,
                              min_child_weight=1,
                              gamma=0,
                              subsample=0.7,
                              colsample_bytree=0.7,
                              reg_alpha=0.005,
                              random_state=42,
                              n_jobs=-1)
xgbr_model.fit(X_train, y_train)
print("Treinamento do XGBoost completo.")

# --- 4. Realizar Previsões ---

# Obtém previsões de ambos os modelos
rf_predictions = rf_model.predict(X_test)
xgbr_predictions = xgbr_model.predict(X_test)

# Ensemble Simples: Média das previsões
# Você também poderia usar média ponderada ou stacking para melhores resultados
ensemble_predictions = (rf_predictions + xgbr_predictions) / 2

# Transforma inversamente as previsões da escala logarítmica de volta para a escala original
final_predictions = np.expm1(ensemble_predictions)

# Garante que não haja previsões negativas (preços não podem ser negativos)
final_predictions[final_predictions < 0] = 0


# --- 5. Criar Arquivo de Submissão ---
submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': final_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nArquivo de submissão 'submission.csv' criado com sucesso! 🎉")
print(submission_df.head())
