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
# # Целевая переменная - точки входа в long. Развороты на графике

# %% [markdown]
# **Назначение:** Функция **annotate_longs_mod** идентифицирует и фильтрует точки входа (buy) и выхода (sell) на основе анализа локальных минимумов и максимумов ценового графика.\
# Ключевые особенности:
# - Поиск локальных экстремумов в скользящем окне
# - Многоуровневая фильтрация сигналов
#
# Параметры:
# - **win_size** - размер окна для поиска экстремумов
# - **min_margin** - минимальное движение внутри окна
# - **min_price_distance** - минимальное расстояние в цене (в процентах) между сигналами
# - **target_candles** — количество свечей для проверки достижения целевого движения
#
# Возвращает: DataFrame с колонками:
# - **buy_noised, sell_noised** — базовые шумные сигналы
# - **buy, sell** — отфильтрованные сигналы (с проверкой роста/падения)
# - **buy_strong, sell_strong** — сильные отфильтрованные сигналы (с дополнительной фильтрацией расстояния)

# %% [markdown]
# ## annotate_longs_mod

# %%
def annotate_longs_mod(
    df,
    win_size=15,
    min_margin=0.005,
    min_price_distance=0.01,
    target_candles=30
):
    """
    Аннотирует локальные минимумы и максимумы для торговых сигналов.

    Args:
        df: DataFrame с колонками 'Close', 'High', 'Low'
        win_size: размер окна для поиска экстремумов
        min_margin: минимальная разница между максимумом и минимумом
        min_price_distance: минимальное расстояние в цене для фильтра (в процентах)
        target_candles: количество свечей для проверки роста после сигнала

    Returns:
        DataFrame с колонками:
        - buy, sell: базовые шумные сигналы
        - buy_filtered, sell_filtered: отфильтрованные сигналы (рост после buy)
        - buy_strong, sell_strong: сильные отфильтрованные сигналы
    """
    tdf = df.copy()
    n = len(tdf)

    # Инициализация массивов для базовых шумных данных
    buy_noised = np.zeros(n)
    sell_noised = np.zeros(n)

    # Векторизованный поиск экстремумов
    for i in range(n - win_size + 1):
        window = tdf.iloc[i:i + win_size]
        close_vals = window['Close'].values

        # Находим индексы минимума и максимума в окне
        min_idx = np.argmin(close_vals)
        max_idx = np.argmax(close_vals)

        # Проверяем условия для минимума
        if min_idx not in [0, win_size - 1]:
            actual_min_idx = window.index[min_idx]
            buy_noised[actual_min_idx] = 1

        # Проверяем условия для максимума
        if max_idx not in [0, win_size - 1]:
            actual_max_idx = window.index[max_idx]
            sell_noised[actual_max_idx] = 1

    # Применяем фильтр min_margin
    for i in range(n):
        if buy_noised[i] == 1 or sell_noised[i] == 1:
            start_idx = max(0, i - win_size + 1)
            end_idx = min(n, i + win_size)
            window = tdf.iloc[start_idx:end_idx]

            min_val = window['Close'].min()
            max_val = window['Close'].max()

            if max_val - min_val <= max_val * min_margin:
                if buy_noised[i] == 1:
                    buy_noised[i] = 0
                if sell_noised[i] == 1:
                    sell_noised[i] = 0

    # Добавляем базовые шумные столбцы
    tdf['buy_noised'] = buy_noised
    tdf['sell_noised'] = sell_noised

    # Применяем фильтр роста цены для buy сигналов
    buy_filtered = np.zeros(n)
    high_prices = tdf['High'].values
    for i in range(n):
        if buy_noised[i] == 1:
            entry_price = tdf['Close'].iloc[i]
            target_price = entry_price * (1 + min_price_distance)

            window_end = min(i + target_candles + 1, n)
            for j in range(i + 1, window_end):
                if high_prices[j] >= target_price:
                    buy_filtered[i] = 1
                    break

    # Применяем фильтр падения цены для sell сигналов
    sell_filtered = np.zeros(n)
    low_prices = tdf['Low'].values
    for i in range(n):
        if sell_noised[i] == 1:
            entry_price = tdf['Close'].iloc[i]
            target_price = entry_price * (1 - min_price_distance)

            window_end = min(i + target_candles + 1, n)
            for j in range(i + 1, window_end):
                if low_prices[j] <= target_price:
                    sell_filtered[i] = 1
                    break

    # Добавляем отфильтрованные столбцы
    tdf['buy'] = buy_filtered
    tdf['sell'] = sell_filtered

    # Создаем сигнальный столбец для сильной фильтрации
    sig = buy_noised - sell_noised
    sig_df = pd.DataFrame({'sig': sig}, index=tdf.index)
    sig_nonzero = sig_df[sig != 0].copy()

    sig_nonzero['flip'] = sig_nonzero['sig'] != sig_nonzero['sig'].shift(1)
    sig_nonzero.iloc[0, sig_nonzero.columns.get_loc('flip')] = True

    # Фильтрация сигналов для сильных точек
    buy_strong = np.zeros(n)
    sell_strong = np.zeros(n)

    current_buy_indices = []
    current_buy_values = []
    current_sell_indices = []
    current_sell_values = []
    last_buy_price = None
    last_sell_price = None

    for idx, row in sig_nonzero.iterrows():
        sig_val = row['sig']
        is_flip = row['flip']
        current_price = tdf.loc[idx, 'Close']

        if sig_val == 1:  # buy signal
            if is_flip:
                current_buy_indices = []
                current_buy_values = []

            if last_sell_price is not None:
                price_diff = (last_sell_price - current_price) / last_sell_price
                if price_diff < min_price_distance:
                    continue

            current_buy_indices.append(idx)
            current_buy_values.append(current_price)

        elif sig_val == -1:  # sell signal
            if is_flip:
                if current_buy_indices:
                    best_buy_idx = current_buy_indices[np.argmin(current_buy_values)]
                    buy_price = tdf.loc[best_buy_idx, 'Close']

                    if last_buy_price is None or (current_price - buy_price) / buy_price >= min_price_distance:
                        buy_strong[tdf.index.get_loc(best_buy_idx)] = 1
                        last_buy_price = buy_price

                current_sell_indices = []
                current_sell_values = []

            if last_buy_price is not None:
                price_diff = (current_price - last_buy_price) / last_buy_price
                if price_diff < min_price_distance:
                    continue

            current_sell_indices.append(idx)
            current_sell_values.append(current_price)

            if is_flip and current_sell_indices:
                best_sell_idx = current_sell_indices[np.argmax(current_sell_values)]
                sell_price = tdf.loc[best_sell_idx, 'Close']

                if last_buy_price is None or (sell_price - last_buy_price) / last_buy_price >= min_price_distance:
                    sell_strong[tdf.index.get_loc(best_sell_idx)] = 1
                    last_sell_price = sell_price

    # Обрабатываем оставшиеся группы
    if current_buy_indices:
        best_buy_idx = current_buy_indices[np.argmin(current_buy_values)]
        buy_price = tdf.loc[best_buy_idx, 'Close']

        if last_sell_price is None or (last_sell_price - buy_price) / last_sell_price >= min_price_distance:
            buy_strong[tdf.index.get_loc(best_buy_idx)] = 1

    if current_sell_indices:
        best_sell_idx = current_sell_indices[np.argmax(current_sell_values)]
        sell_price = tdf.loc[best_sell_idx, 'Close']

        if last_buy_price is None or (sell_price - last_buy_price) / last_buy_price >= min_price_distance:
            sell_strong[tdf.index.get_loc(best_sell_idx)] = 1

    # Добавляем сильные сигналы в DataFrame
    tdf['buy_strong'] = buy_strong
    tdf['sell_strong'] = sell_strong

    return tdf


# %% [markdown]
# ## annotate_longs_mod_rr_threshold

# %% [markdown]
# Аналог annotate_longs_mod но с добавлением соотношения риск/прибыль

# %%
def annotate_longs_mod_rr_threshold(df, win_size=15, min_margin=0.005, min_price_distance=0.01,
                       target_candles=30, rr_threshold=2.0):
    """
    Исправленная версия: маркирует buy=1 только если TP случился раньше SL.
    Возвращает DataFrame с колонками:
      - buy_noised, sell_noised  (сырые экстремумы)
      - buy, sell                (фильтр: TP before SL на горизонте target_candles)
      - buy_strong, sell_strong  (доп. фильтрация по группам/разстояниям)
    """
    tdf = df.copy()
    n = len(tdf)

    # ------- базовые Noised сигналы: локальные min/max в окне -------
    buy_noised = np.zeros(n, dtype=int)
    sell_noised = np.zeros(n, dtype=int)

    for i in range(0, n - win_size + 1):
        window = tdf.iloc[i:i + win_size]
        close_vals = window['Close'].values

        min_idx = int(np.argmin(close_vals))
        max_idx = int(np.argmax(close_vals))

        # мин и макс не на краях окна
        if min_idx not in (0, win_size - 1):
            actual_min_label = window.index[min_idx]
            pos = tdf.index.get_loc(actual_min_label)
            buy_noised[pos] = 1

        if max_idx not in (0, win_size - 1):
            actual_max_label = window.index[max_idx]
            pos = tdf.index.get_loc(actual_max_label)
            sell_noised[pos] = 1

    # ------- фильтр min_margin: убираем слишком мелкие экстремумы -------
    for i in range(n):
        if buy_noised[i] == 1 or sell_noised[i] == 1:
            start_idx = max(0, i - win_size + 1)
            end_idx = min(n, i + win_size)
            window = tdf.iloc[start_idx:end_idx]
            min_val = window['Close'].min()
            max_val = window['Close'].max()
            if (max_val - min_val) <= max_val * min_margin:
                buy_noised[i] = 0
                sell_noised[i] = 0

    tdf['buy_noised'] = buy_noised
    tdf['sell_noised'] = sell_noised

    # ------- вычисление TP/SL (для long и short) -------
    close = tdf['Close'].values
    high = tdf['High'].values
    low = tdf['Low'].values

    sl_pct = min_price_distance / rr_threshold  # SL = TP / rr

    buy_filtered = np.zeros(n, dtype=int)
    sell_filtered = np.zeros(n, dtype=int)

    for i in range(n):
        # LONG candidate
        if buy_noised[i] == 1:
            entry = close[i]
            tp_price = entry * (1 + min_price_distance)
            sl_price = entry * (1 - sl_pct)

            window_end = min(i + target_candles + 1, n)
            tp_hit_first = False

            for j in range(i + 1, window_end):
                hit_sl = low[j] <= sl_price
                hit_tp = high[j] >= tp_price

                if hit_sl and hit_tp:
                    # SL и TP на одной свече → SL считается первым → no signal
                    tp_hit_first = False
                    break
                elif hit_sl:
                    tp_hit_first = False
                    break
                elif hit_tp:
                    tp_hit_first = True
                    break

            if tp_hit_first:
                buy_filtered[i] = 1

        # SHORT candidate
        if sell_noised[i] == 1:
            entry = close[i]
            tp_price_short = entry * (1 - min_price_distance)  # target down
            sl_price_short = entry * (1 + sl_pct)             # stop up

            window_end = min(i + target_candles + 1, n)
            tp_hit_first_short = False

            for j in range(i + 1, window_end):
                hit_sl_short = high[j] >= sl_price_short   # stop for short
                hit_tp_short = low[j] <= tp_price_short    # tp for short

                if hit_sl_short and hit_tp_short:
                    # SL и TP на одной свече → SL считается первым → no signal
                    tp_hit_first_short = False
                    break
                elif hit_sl_short:
                    tp_hit_first_short = False
                    break
                elif hit_tp_short:
                    tp_hit_first_short = True
                    break

            if tp_hit_first_short:
                sell_filtered[i] = 1

    tdf['buy'] = buy_filtered
    tdf['sell'] = sell_filtered

    # ------- подсильные сигналы (та же логика, что и у тебя, не менял сильно) -------
    sig = buy_noised - sell_noised
    sig_df = pd.DataFrame({'sig': sig}, index=tdf.index)
    sig_nonzero = sig_df[sig != 0].copy()

    # отмечаем первые как flip=True
    sig_nonzero['flip'] = (sig_nonzero['sig'] != sig_nonzero['sig'].shift(1))
    if len(sig_nonzero) > 0:
        sig_nonzero.iloc[0, sig_nonzero.columns.get_loc('flip')] = True

    buy_strong = np.zeros(n, dtype=int)
    sell_strong = np.zeros(n, dtype=int)

    current_buy_indices = []
    current_buy_values = []
    current_sell_indices = []
    current_sell_values = []

    last_buy_price = None
    last_sell_price = None

    for idx, row in sig_nonzero.iterrows():
        sig_val = int(row['sig'])
        is_flip = bool(row['flip'])
        current_price = float(tdf.loc[idx, 'Close'])

        if sig_val == 1:  # buy signal
            if is_flip:
                current_buy_indices = []
                current_buy_values = []

            # проверяем расстояние от последнего sell
            if last_sell_price is not None:
                price_diff = (last_sell_price - current_price) / last_sell_price
                if price_diff < min_price_distance:
                    continue

            current_buy_indices.append(idx)
            current_buy_values.append(current_price)

        elif sig_val == -1:  # sell signal
            if is_flip:
                # обработать предыдущую группу buy
                if current_buy_indices:
                    best_buy_label = current_buy_indices[int(np.argmin(current_buy_values))]
                    best_buy_pos = tdf.index.get_loc(best_buy_label)
                    buy_price = tdf.iloc[best_buy_pos]['Close']

                    # проверяем расстояние от последнего buy
                    if last_buy_price is None or (current_price - buy_price) / buy_price >= min_price_distance:
                        buy_strong[best_buy_pos] = 1
                        last_buy_price = buy_price

                current_sell_indices = []
                current_sell_values = []

            # проверяем расстояние от последнего buy
            if last_buy_price is not None:
                price_diff = (current_price - last_buy_price) / last_buy_price
                if price_diff < min_price_distance:
                    continue

            current_sell_indices.append(idx)
            current_sell_values.append(current_price)

            # если разворот обратно к покупке
            if is_flip and current_sell_indices:
                best_sell_label = current_sell_indices[int(np.argmax(current_sell_values))]
                best_sell_pos = tdf.index.get_loc(best_sell_label)
                sell_price = tdf.iloc[best_sell_pos]['Close']
                if last_buy_price is None or (sell_price - last_buy_price) / last_buy_price >= min_price_distance:
                    sell_strong[best_sell_pos] = 1
                    last_sell_price = sell_price

    # обрабатываем хвостовые группы
    if current_buy_indices:
        best_buy_label = current_buy_indices[int(np.argmin(current_buy_values))]
        best_buy_pos = tdf.index.get_loc(best_buy_label)
        buy_price = tdf.iloc[best_buy_pos]['Close']
        if last_sell_price is None or (last_sell_price - buy_price) / last_sell_price >= min_price_distance:
            buy_strong[best_buy_pos] = 1

    if current_sell_indices:
        best_sell_label = current_sell_indices[int(np.argmax(current_sell_values))]
        best_sell_pos = tdf.index.get_loc(best_sell_label)
        sell_price = tdf.iloc[best_sell_pos]['Close']
        if last_buy_price is None or (sell_price - last_buy_price) / last_buy_price >= min_price_distance:
            sell_strong[best_sell_pos] = 1

    tdf['buy_strong'] = buy_strong
    tdf['sell_strong'] = sell_strong

    return tdf

# %%
