import os
import pandas as pd
from pathlib import Path

def create_monthly_silver_view(
    input_path: Path, 
    output_path: Path, 
    date_col: str = "data", 
    columns_to_drop: list = None
):
    """
    Lê um arquivo CSV da camada bronze, remove colunas indesejadas se especificado,
    calcula métricas mensais e salva a visão na camada silver.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo bronze não encontrado em: {input_path}")

    print(f"Lendo dados de: {input_path}")
    df = pd.read_csv(input_path)
    
    if columns_to_drop:
        existing_cols = [col for col in columns_to_drop if col in df.columns]
        if existing_cols:
            df.drop(columns=existing_cols, inplace=True)
            print(f"   -> Colunas removidas: {existing_cols}")
    # --------------------------------------------------------

    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    
    monthly_grouped = df[numeric_cols].resample("ME")

    df_mean = monthly_grouped.mean()
    df_std = monthly_grouped.std()
    df_pct_change = df_mean.pct_change()

    df_mean = df_mean.add_suffix("_mean")
    df_std = df_std.add_suffix("_std")
    df_pct_change = df_pct_change.add_suffix("_pct_change")

    df_silver = pd.concat([df_mean, df_std, df_pct_change], axis=1)
    df_silver = df_silver.reindex(sorted(df_silver.columns), axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_silver.to_parquet(output_path, index=True)

    print(f"Visão mensal salva com sucesso em: {output_path}")
    print(f"Formato final: {df_silver.shape[1]} colunas geradas.\n")

    return df_silver

def build_path(read_data: str, path_final: str):
    current_file = Path(__file__).resolve()
    BASE = current_file.parents[2]   
    
    bronze_archive = BASE / read_data
    silver_dir = BASE / path_final
    
    return bronze_archive, silver_dir

if __name__ == "__main__":
    
    list_of_files = [
        ("data/bronze/dados_bacen.csv", "data/silver/macro_monthly.parquet", None),
        ("data/bronze/markets.csv", "data/silver/markets_monthly.parquet", None),
        ("data/bronze/dados_ipea.csv", "data/silver/ipea_monthly.parquet", ["YEAR", "MONTH"])
    ]

    for read_data, path_final, columns_deleted in list_of_files:
        bronze_archive, silver_dir = build_path(read_data, path_final)
        create_monthly_silver_view(bronze_archive, silver_dir, columns_to_drop=columns_deleted)