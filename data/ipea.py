# %% inflacao, desemprego, pib

import ipeadatapy as ipd
import pandas as pd
import matplotlib.pyplot as plt

def search_series(keyword):
    """Busca por séries relacionadas a um tema específico no IPEA Data. Dados sempre Macro"""
    df = ipd.list_series()
    ipd.list_series().columns
    ipd.metadata(big_theme='Macroeconômico', country='Brasil', frequency='Mensal')
    return df[df['NAME'].str.contains(keyword, case=False)]

inflacao = (search_series('Inflação'))
pib = (search_series('PIB'))
desemprego = (search_series('Desemprego'))

# %%
print(desemprego[['CODE', 'NAME']])
print(inflacao[['CODE', 'NAME']])
print(pib[['CODE', 'NAME']])
# %%
#ipd.timeseries('BM12_IPCAEXP1212') # inflacao
#ipd.timeseries('BM12_PIB12') #pib mensal
ipd.timeseries('PNADC12_TDESOC12') # desemprego
# %%

series_dict = {
    'pib_mensal': 'BM12_PIB12',
    'desemprego_mensal': 'PNADC12_TDESOC12',
    'inflacao_mensal': 'BM12_IPCAEXP1212' # expactativa mensal acumulada 12 meses do IPCA
}


def get_data_ipea(series_dict, start_year):

    data_frames = []
    start_year = start_year - 1
    for name, code in series_dict.items():
        df = ipd.timeseries(code, yearGreaterThan=start_year)
        df = df.reset_index()
        value_col = next(
            col for col in df.columns
            if col.startswith('VALUE')
        )

        df = df[['YEAR', 'MONTH', value_col]].copy()
        df.rename(columns={value_col: name}, inplace=True)
        data_frames.append(df)

    merged_df = data_frames[0]

    for df in data_frames[1:]:

        merged_df = pd.merge(
            merged_df,
            df,
            on=['YEAR', 'MONTH'],
            how='outer'
        )

    merged_df = merged_df.sort_values(
        ['YEAR', 'MONTH']
    ).reset_index(drop=True)

    merged_df['DATE'] = pd.to_datetime(
        dict(
            year=merged_df['YEAR'],
            month=merged_df['MONTH'],
            day=1
        )
    )
    merged_df = merged_df.set_index('DATE')

    return merged_df
# %%

cols = [
    'pib_mensal',
    'desemprego_mensal',
    'inflacao_mensal'
]

def search_missing_values_plot(df, cols):# transforma NaN em 1 e valores existentes em 0
    missing_data = df[cols].isna().astype(int)

    # plota
    missing_data.plot(
        figsize=(14,4)
    )

    plt.title('Valores Nulos nas Séries Macroeconômicas')
    plt.xlabel('Data')
    plt.ylabel('Nulo = 1')
    plt.show()

def plot_series(code, cols, title, ylabel, xlabel='Data'):
    df = ipd.timeseries(code, yearGreaterThan=1995)
    value_col = [col for col in df.columns if col.startswith('VALUE')][0]

    plt.plot(df[value_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

# %%
data = get_data_ipea(series_dict, start_year=2000)
data.head()
# %%
data.to_csv('../db/dados_ipea.csv', index=True)
data.tail(10)
# %%
