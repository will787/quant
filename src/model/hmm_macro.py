# %%
from hmmlearn import hmm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import yfinance as yf

def clean_yfinance_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    return data['Close']


def prepare_features(prices, window_vol=21, window_mom=63):
    """"
        Criação de features ortogonais para HMM detectar os regimes de mercado.
    """
    feature = pd.DataFrame(prices)

    feature.columns = ['price']

    feature['returns'] = np.log(feature['price'] / feature['price'].shift(1))

    feature['volatily'] = feature['returns'].rolling(window_vol).std()

    feature['momentum'] = (feature['price'] / feature['price'].rolling(window_mom).mean()) - 1

    feature['z_score'] =  (feature['price'] - feature['price'].rolling(window_vol).mean()) / feature['price'].rolling(window_vol).std() #* np.sqrt(window_vol)

    feature['vol_regime'] = feature['volatily'] / feature['volatily'].rolling(252).mean()

    return feature.dropna()

p_ibov = clean_yfinance_data('^BVSP', '2000-01-01', '2026-06-01')

df_features = prepare_features(p_ibov)

columns_hmm = ['returns', 'volatily', 'momentum', 'z_score', 'vol_regime']


X_treino = df_features[columns_hmm].values

modelo_macro = hmm.GaussianHMM(n_components=5, covariance_type='full', n_iter=100, random_state=42)
modelo_macro.fit(X_treino)

idx_vol = 1

vol_states = [modelo_macro.means_[i][idx_vol] for i in range(modelo_macro.n_components)]

order_states_por_risco = np.argsort(vol_states)

dicionario_regimes = {
    order_states_por_risco[0]: 'Bull market (Baixa Volatilidade)',
    order_states_por_risco[1]: 'Bull Normal (Volatilidade Moderada)',
    order_states_por_risco[2]: 'Lateral indeciso',
    order_states_por_risco[3]: 'Correção (Alta Volatilidade)',
    order_states_por_risco[4]: 'Crash / Pânico (Muito Alta Volatilidade)'
}


#previsao
estado_previstos = modelo_macro.predict(X_treino)
df_features['regimes'] = estado_previstos

#df_features['regimes'] = df_features['regimes'].map(dicionario_regimes)
df_features['regimes'] = df_features['regimes'].astype(str)

fig = px.scatter(df_features, x=df_features.index, y='price', color='regimes', 
                 title='S&P 500 / Ibovespa - Regimes de Mercado (HMM)',
                 log_y=True) # Escala logarítmica é melhor para prazos longos
fig.update_traces(marker=dict(size=3)) # Diminui o tamanho das bolinhas para ver melhor a linha
fig.show()

transmat = modelo_macro.transmat_
print("\nMatriz de Transição (Para o Bellman):")
print(transmat)

df_features.head()
# %%

colunas_hmm = ['returns', 'volatily', 'momentum', 'z_score', 'vol_regime']

df_features.index = pd.to_datetime(df_features.index)

ano_inicio_treino = 2000
ano_fim_primeiro_treino = 2004 
ano_fim_dados = df_features.index.year.max()

previsoes_out_of_sample = pd.Series(dtype=int)


print(f"Iniciando Walk-Forward (Blocos de 2 anos). Treino base até {ano_fim_primeiro_treino}...")

for ano_teste_inicio in range(ano_fim_primeiro_treino + 1, ano_fim_dados + 1, 2):
    
    ano_teste_fim = ano_teste_inicio + 1

    dados_treino = df_features[df_features.index.year <= ano_teste_inicio]
    
    mascara_tese = (df_features.index.year >= ano_teste_inicio) & (df_features.index.year <= ano_teste_fim)
    dados_teste = df_features[mascara_tese]

    if dados_teste.empty:
        break

    X_treino = dados_treino[colunas_hmm].values
    X_teste = dados_teste[colunas_hmm].values

    modelo = hmm.GaussianHMM(n_components=5, covariance_type='full', n_iter=100, random_state=42)
    modelo.fit(X_treino)

    diagonal_media = np.mean(modelo.transmat_)
    status_overfit = "Alerta" if diagonal_media < 0.8 else "Estável OK"

    idx_vol = 1
    volatilidades = [modelo.means_[i][idx_vol] for i in range(modelo.n_components)]
    ordem_risco = np.argsort(volatilidades)
    mapa_risco = {estado_hmm: nivel_risco for nivel_risco, estado_hmm in enumerate(ordem_risco)}
    
    estados_crus = modelo.predict(X_teste)
    estados_corrigidos = [mapa_risco[estado] for estado in estados_crus]

    previsoes_out_of_sample = pd.concat([
        previsoes_out_of_sample,
        pd.Series(estados_corrigidos, index=dados_teste.index)
    ])
    
    print(f"Bloco {ano_teste_inicio}-{ano_teste_fim} concluído | "
          f"Persistência Média: {diagonal_media*100:.1f}% | {status_overfit}")


# --- INTEGRAÇÃO FINAL ---
df_final = df_features.copy()
df_final['Regime_WalkForward'] = previsoes_out_of_sample

df_operavel = df_final.dropna(subset=['Regime_WalkForward']).copy()

# Dicionário e Plotagem (Igual ao anterior)
dicionario_regimes = {
    0: "1. Bull Fortíssimo",
    1: "2. Bull Normal",
    2: "3. Lateral / Indeciso",
    3: "4. Correção",
    4: "5. Crash / Pânico"
}

df_operavel['Nome_Regime'] = df_operavel['Regime_WalkForward'].map(dicionario_regimes)

fig = px.scatter(df_operavel, x=df_operavel.index, y='price', color='Nome_Regime', 
                 title='Ibovespa - Walk-Forward de 2 em 2 Anos (Out-of-Sample)',
                 log_y=True,
                 category_orders={"Nome_Regime": list(dicionario_regimes.values())})

fig.update_traces(marker=dict(size=3))
fig.show()
# %%
