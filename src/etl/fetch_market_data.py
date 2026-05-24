# %%
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analisar_correlacoes_globais(df):
    """
    Análise de correlações entre mercados
    """
    retornos = df.pct_change().dropna()
    
    # Matriz de correlações
    plt.figure(figsize=(10,8))
    sns.heatmap(retornos.corr(method='spearman'), annot=True, cmap='RdBu_r', center=0)
    plt.title('Correlações Entre Mercados Globais')
    plt.tight_layout()
    plt.show()
    
    # Estatísticas descritivas
    print("=== CORRELAÇÕES COM IBOVESPA ===")
    corr_ibovespa = retornos.corr(method='spearman')['ibovespa_br'].sort_values(ascending=False)
    print(corr_ibovespa)
    
    return retornos

def cross_correlations(df, lags):

    retornos = df.pct_change().dropna()

    mercados = retornos.columns.tolist()

    resultados = {}

    for mercado1 in mercados:

        resultados[mercado1] = {}

        for mercado2 in mercados:

            if mercado1 != mercado2:

                correlacoes = []

                for lag in range(-lags, lags + 1):

                    if lag == 0:

                        corr = retornos[mercado1].corr(
                            retornos[mercado2],
                            method='spearman'
                        )

                    elif lag > 0:

                        corr = retornos[mercado1].corr(
                            retornos[mercado2].shift(lag),
                            method='spearman'
                        )

                    else:

                        corr = retornos[mercado1].shift(-lag).corr(
                            retornos[mercado2],
                            method='spearman'
                        )

                    correlacoes.append(corr)

                resultados[mercado1][mercado2] = correlacoes

    return resultados

def get_dy_stocks(tickers, start_date, end_date):
    return None

def get_finance_data(tickers, start_date, end_date):

    data_frames = []

    for ticker in tickers:

        try:

            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )

            # pega fechamento
            close_prices = df['Close']

            # renomeia série
            close_prices.name = ticker

            print(f"{ticker}: {len(close_prices)} observações.")

            data_frames.append(close_prices)

        except Exception as e:

            print(f"Erro ao baixar dados para {ticker}: {e}")

    if data_frames:

        merged_data = pd.concat(
            data_frames,
            axis=1
        )

        return merged_data

    else:

        print("Nenhum dado foi baixado.")

        return pd.DataFrame()


finance_data = get_finance_data(
    ['^BVSP', '^GSPC', '^GDAXI', '000001.SS', '^N225', '^MERV', '^FTSE'],
    '2000-01-01',
    '2026-03-31'
)

df = finance_data.rename(columns={
    '^BVSP': 'ibovespa_br',
    '^GSPC': 's&p500_eua',
    '^GDAXI': 'dax_alemanha',
    '000001.SS': 'shanghai_china',
    '^N225': 'nikkei_japao',
    '^MERV': 'merv_argentina',
    '^FTSE': 'ftse_uk',
    })

print(df.head())
df.to_csv('../db/markets.csv', index=True)

ret = analisar_correlacoes_globais(df)
# %%


cross_corr = cross_correlations(df, lags=3)

lags = range(-3, 4)

corrs = cross_corr['shanghai_china']['s&p500_eua']
corrs = np.array(corrs)

best_lag = lags[np.argmax(np.abs(corrs))]

best_corr = corrs[np.argmax(np.abs(corrs))]

print(best_lag)
print(best_corr)
plt.figure(figsize=(10,4))

plt.plot(lags, corrs)

plt.axhline(0, linestyle='--')
plt.axvline(0, linestyle='--')

plt.title('Cross-Correlation: Shanghai vs S&P 500')
plt.xlabel('Lag')
plt.ylabel('Correlação')

plt.show()
# %%
