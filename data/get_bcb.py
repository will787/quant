# %%
import pandas as pd
from bcb import sgs

def coletar_fluxo_capitais(start_date='2020-01-01'):
    series = {
        'selic': 432,
        'ied_liquido': 2860,
        'dolar_cambio_livre': 1, #venda (quanto esta vendendo para comprar real)
        'euro_cambio_livre': 21619,
        'iene_cambio_livre': 21621
        
    }
    dados_fluxo = {}
    for nome, codigo in series.items():
        try:
            serie = sgs.get(codigo, start=start_date)
            serie = serie.rename(columns={codigo: nome})
            
            print(f"{nome}:")
            print(f"Período: {serie.index.min()} a {serie.index.max()}")
            print(f"Total de observações: {len(serie)}\n")
            
            dados_fluxo[nome] = serie
        
        except Exception as e:
            print(f"Erro: {e}")
    
    # Combinar séries
    df_fluxo_capitais = pd.concat(dados_fluxo.values(), axis=1)
    return df_fluxo_capitais

# Coletar dados
df_bacen = coletar_fluxo_capitais()

df_bacen = df_bacen.rename(columns={
    '432': 'selic',
    '2860': 'ied_liquido',
    '1': 'dolar_cambio_livre',
    '21619': 'euro_cambio_livre',
    '21621': 'iene_cambio_livre'
})

# %%
df_bacen
# %%
