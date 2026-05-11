# %%
import pandas as pd
from bacendata import sgs #consulta para séries longas
from bcb import Expectativas

def coletar_fluxo_capitais(start_date='2000-01-01', end_date='2026-03-31'):
    series = {
        'meta_taxa_selic': 432,
        'taxa_selic': 11,
        'dolar_cambio_livre': 1, #venda (quanto esta vendendo para comprar real)
        'euro_cambio_livre': 21619,
        'iene_cambio_livre': 21621
        
    }
    dados_fluxo = {}
    for nome, codigo in series.items():
        try:
            serie = sgs.get(codigo, start=start_date, end=end_date)
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
# %%
df_bacen.columns = [
    'meta_taxa_selic',
    'taxa_selic',
    'dolar_cambio_livre_p_tax',
    'euro_cambio_livre',
    'iene_cambio_livre'
]
df_bacen.tail(10)
df_bacen.head()
df_bacen.to_csv('dados_bacen.csv', index=True)
# %%
