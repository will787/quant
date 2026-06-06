# %%
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
from pathlib import Path

def read_gold_data(gold_path: str):
    p = Path(gold_path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo Gold não encontrado em: {gold_path}")
    return pd.read_parquet(p)

def create_targets_and_lags(df: pd.DataFrame, lags=[1,3,6,12]):
    df = df.sort_index()
    df = df.copy()
    df['ibov_ret_1m'] = df['ibovespa_br_pct_change'].shift(-1)
    feature_cols = [c for c in df.columns if c != 'ibov_ret_1m']
    for col in feature_cols:
        for l in lags:
            df[f'{col}_lag{l}'] = df[col].shift(l)
    # dropar apenas linhas sem target (última linha(s))
    df = df.dropna(subset=['ibov_ret_1m'])
    return df

def split_data(df: pd.DataFrame, train_end: str, val_end: str):
    train = df.loc[:train_end].copy()
    val = df.loc[train_end:val_end].copy()
    test = df.loc[val_end:].copy()
    return train, val, test

def fit_preprocessors(train: pd.DataFrame, feature_cols):
    imp = SimpleImputer(strategy='median')
    imp.fit(train[feature_cols])
    sc = RobustScaler()
    sc.fit(imp.transform(train[feature_cols]))
    return imp, sc

def prepare_data(imp, sc, df: pd.DataFrame, feature_cols):
    X = imp.transform(df[feature_cols])
    X = sc.transform(X)
    y = df['ibov_ret_1m'].values
    return X, y

if __name__ == "__main__":
    df = read_gold_data("data/gold/monthly_data.parquet")
    df = create_targets_and_lags(df)
    feature_cols = [c for c in df.columns if c not in ['ibov_ret_1m']]
    train, val, test = split_data(df, "2015-12-31", "2018-12-31")
    imp, sc = fit_preprocessors(train, feature_cols)
    X_train, y_train = prepare_data(imp, sc, train, feature_cols)
    X_val, y_val = prepare_data(imp, sc, val, feature_cols)
    X_test, y_test = prepare_data(imp, sc, test, feature_cols)

    print("Dados preparados:")
    print(X_test)
#
# %%
