# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python [conda env:trading_env]
#     language: python
#     name: conda-env-trading_env-py
# ---

# %%
import numpy as np
import datetime
import time
import numpy as np
import pandas as pd
import pandas_ta as ta


# %% [markdown]
# # Целевая переменная target

# %%
def add_target_column(df, target_candles=20, target=0.04, rr_threshold=2.0):
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    y = np.zeros(len(df), dtype=int)

    sl_pct = target / rr_threshold  # SL = target / rr_threshold

    for i in range(len(df)):
        entry_price = close[i]
        tp_price = entry_price * (1 + target)
        sl_price = entry_price * (1 - sl_pct)

        window_end = min(i + target_candles + 1, len(df))
        tp_hit_first = False

        for j in range(i + 1, window_end):
            hit_sl = low[j] <= sl_price
            hit_tp = high[j] >= tp_price

            if hit_sl and hit_tp: # SL и TP на одной свече → SL считается первым → y=0
                break
            elif hit_sl: # SL раньше → y=0
                break
            elif hit_tp: # TP раньше → y=1
                tp_hit_first = True
                break

        if tp_hit_first:
            y[i] = 1

    df['target'] = y
    return df
#Пример как вызвать
# df = add_target_column_no_overlap(df,target_pct=0.025,target_candles=20,rr_threshold=2.0)


# %% [markdown]
# ## Целевая переменная с фильтром

# %%
def add_target_column_mod(
    df,
    target_candles=20,
    target=0.04,
    rr_threshold=2.0,
    use_macd_filter=False, macd_fast=12, macd_slow=26, macd_signal=9,
    use_sma200_filter=False,
    use_volume_filter=False, vol_ma_len=20
):
    df = df.copy()

    # ===== БАЗА: ровно как в твоей add_target_column =====
    close = df['Close'].values
    high  = df['High'].values
    low   = df['Low'].values
    y = np.zeros(len(df), dtype=int)

    sl_pct = target / rr_threshold  # SL = target / rr_threshold

    for i in range(len(df)):
        entry_price = close[i]
        tp_price = entry_price * (1 + target)
        sl_price = entry_price * (1 - sl_pct)

        window_end = min(i + target_candles + 1, len(df))
        tp_hit_first = False

        for j in range(i + 1, window_end):
            hit_sl = low[j] <= sl_price
            hit_tp = high[j] >= tp_price

            if hit_sl and hit_tp:   # SL и TP на одной свече → SL первый
                break
            elif hit_sl:            # SL раньше → 0
                break
            elif hit_tp:            # TP раньше → 1
                tp_hit_first = True
                break

        if tp_hit_first:
            y[i] = 1

    df['target'] = y

    # ===== ФИЛЬТРЫ/СЕТАПЫ: только маскируют target =====
    setup_ok = pd.Series(True, index=df.index)
    tmp_cols = []

    if use_macd_filter:
        macd_df = ta.macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        macd_col  = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
        macds_col = f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"
        df['_macd']  = macd_df[macd_col]
        df['_macds'] = macd_df[macds_col]
        setup_ok &= df['_macd'] > df['_macds']
        tmp_cols += ['_macd', '_macds']

    if use_sma200_filter:
        df['_sma200'] = ta.sma(df['Close'], length=200)
        setup_ok &= df['Close'] > df['_sma200']
        tmp_cols.append('_sma200')

    if use_volume_filter:
        df['_volma'] = df['Volume'].rolling(vol_ma_len).mean()
        setup_ok &= df['Volume'] > df['_volma']
        tmp_cols.append('_volma')

    # Применяем фильтры к таргету
    df.loc[~setup_ok, 'target'] = 0

    # Чистим временные колонки
    if tmp_cols:
        df.drop(columns=[c for c in tmp_cols if c in df.columns], inplace=True)

    return df


# %%
def add_target_column_simple(df, target_candles=20, target=0.04):
    close = df['Close'].values
    high = df['High'].values
    y = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        entry_price = close[i]
        tp_price = entry_price * (1 + target)  # Цена тейк-профита
        
        # Проверяем следующие target_candles свечей
        window_end = min(i + target_candles + 1, len(df))
        
        for j in range(i + 1, window_end):
            if high[j] >= tp_price:  # Если цена достигла TP
                y[i] = 1
                break

    df['target'] = y
    return df

# %%
