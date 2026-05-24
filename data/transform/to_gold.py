import pandas as pd
from pathlib import Path

def create_monthly_gold_view(
    input_paths: list[Path], 
    output_path: Path, 
    date_col: str = "data", 
    columns_to_drop: list = None
):
    """
    Lê múltiplos arquivos da camada Silver (já agregados),
    combina todos pelo índice de data e salva o dataset final unificado na camada Gold.
    """
    print(f"Iniciando a consolidação da camada GOLD...")
    silvers_dfs = []

    # 1. Carregar e acumular todos os arquivos Silver passados
    for path in input_paths:
        if not path.exists():
            print(f"   [Aviso] Arquivo Silver não encontrado: {path}. Pulando...")
            continue
            
        print(f"   -> Lendo e alinhando: {path.name}")
        # Como o índice já é a data do resample, lemos mantendo o índice
        df_silver = pd.read_parquet(path)
        silvers_dfs.append(df_silver)

    if not silvers_dfs:
        raise ValueError("Nenhum arquivo Silver válido foi encontrado para consolidação.")

    # 2. Concatenar horizontalmente (axis=1) alinhando pelo índice (data)
    # join="outer" garante que nenhuma data seja perdida (se houver gaps, preenche com NaN)
    df_gold = pd.concat(silvers_dfs, axis=1, join="outer")

    # 3. Remover colunas indesejadas na Gold se necessário
    if columns_to_drop:
        existing_cols = [col for col in columns_to_drop if col in df_gold.columns]
        if existing_cols:
            df_gold.drop(columns=existing_cols, inplace=True)
            print(f"   -> Colunas removidas na Gold: {existing_cols}")

    # 4. Ordenar colunas para manter o DataFrame limpo
    df_gold = df_gold.reindex(sorted(df_gold.columns), axis=1)

    # 5. Salvar o resultado final na pasta Gold
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_gold.to_parquet(output_path, index=True)

    print(f"\n[Sucesso] Camada GOLD criada e salva em: {output_path}")
    print(f"Dimensões finais da Gold: {df_gold.shape[0]} meses X {df_gold.shape[1]} colunas.\n")

    return df_gold

def build_path(read_data: str, path_final: str):
    current_file = Path(__file__).resolve()
    BASE = current_file.parents[2]   # quant/
    
    return BASE / read_data, BASE / path_final

if __name__ == "__main__":
    # Lista de arquivos Silver a serem consolidados na Gold
    silver_files = [
        "data/silver/macro_monthly.parquet",
        "data/silver/markets_monthly.parquet",
        "data/silver/ipea_monthly.parquet"
    ]
    
    silver_paths = [Path(__file__).resolve().parents[2] / path for path in silver_files]
    gold_output_path = Path(__file__).resolve().parents[2] / "data/gold/monthly_data.parquet"

    create_monthly_gold_view(silver_paths, gold_output_path)
    