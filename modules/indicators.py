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

# %% [markdown]
# # Импорт библиотек

# %%
import numpy as np
import pandas as pd
import pandas_ta as ta

# %%
eps = 1e-8


# %% [markdown]
# # EMA

# %% [markdown]
# ## delta_ema

# %% [markdown]
# 📌 Описание функции **delta_ema** *(df, ema_lenght=[20], block_size=5, shifts=None)*\
# **Вход:**
# - **df** — DataFrame с историческими свечами и обязательной колонкой Close.
# - **ema_lengths** — список длин EMA, например [20, 50].
# - **window_size** — общее кол-во свечей для анализа
# - **num_blocks** — на сколько частей делим окно
#
# **Что делает:**\
# Для каждой EMA из списка **ema_lenghts**:
#
# Считает относительное изменение EMA внутри каждого блока:
# $$\frac{ema_{start} - ema_{end}}{ema_{start}}$$
# Определяет, в скольких блоках EMA росла (значения в диапазоне 0–4 при стандартных настройках).\
# **Выход:**\
# Возвращает тот же df, но с добавленными колонками:\
# block1_emaXX_rel_change, ..., block4_emaXX_rel_change — относительные изменения EMA для каждого блока.\
# emaXX_grow_blocks — количество растущих блоков для данной EMA.

# %%
def delta_ema(df, ema_lengths=[20], window_size=20, num_blocks=4):
    close_norn = df['Close'] / df['Close'].rolling(20).max()
    block_size = window_size // num_blocks
    shifts = [block_size * i for i in range(1, num_blocks + 1)]

    df = df.copy()

    for ema_length in ema_lengths:
        ema = ta.ema(close_norn, ema_length)

        # Изменения EMA в каждом блоке
        for i, shift in enumerate(shifts, 1):
            start_vals = ema.shift(shift - block_size)
            end_vals = ema.shift(shift)
            df[f'block{i}_ema{ema_length}_rel_change'] = (end_vals - start_vals) / start_vals

        # Кол-во растущих блоков
        grow_flags = [(ema.shift(s) > ema.shift(s - block_size)) for s in shifts]
        df[f'ema{ema_length}_grow_blocks'] = pd.concat(grow_flags, axis=1).sum(axis=1).astype(int)

    return df


# %% [markdown]
# ## ema_above_price

# %% [markdown]
# Добавляет бинарные признаки, показывающие, находится ли EMA выше цены закрытия

# %%
def ema_above_price(df, lengths=[20, 100]):
    df = df.copy()
    
    for length in lengths:
        # Вычисляем EMA с помощью pandas_ta
        ema = ta.ema(df['Close'], length=length)
        
        # Создаем бинарный признак (1 - EMA выше цены, 0 - нет)
        df[f'ema{length}_above_price'] = (ema > df['Close']).astype(int)
    
    return df


# %% [markdown]
# ## ema_speed

# %% [markdown]
# Добавляет признаки скорости и ускорения EMA с нормировкой.\
# Входные данные:
# - Периоды EMA для расчета (по умолчанию [20, 50])
# - Окна для расчета скорости/ускорения (по умолчанию [10, 30, 60])
#
# На выходе получаем первую и вторую производную ema

# %%
def ema_speed(df, lengths=[20, 50], windows=[10, 30, 60]):

    df = df.copy()
    
    for length in lengths:
        # Рассчитываем EMA во временной переменной
        ema_values = ta.ema(df['Close'], length=length)
        
        for window in windows:
            # Скорость = среднее изменение EMA за window свечей
            speed_col = f'ema{length}_speed_{window}'
            speed_values = (ema_values - ema_values.shift(window)) / window
            
            # Нормировка скорости на SMA цены за тот же период
            norm_factor = df['Close'].rolling(window).mean()
            df[speed_col] = speed_values / norm_factor
            
            # Ускорение = изменение скорости
            accel_col = f'ema{length}_accel_{window}'
            df[accel_col] = df[speed_col].diff(window) / window
            
            # Нормировка ускорения
            df[accel_col] = df[accel_col] / norm_factor
            
    return df


# %% [markdown]
# # MACD

# %% [markdown]
# ## macd_cross

# %% [markdown]
# 📌 Описание функции: **macd_cross** *(df, fast=12, slow=26, signal=9)*
#
# Вход:
# - **df** — DataFrame с колонкой Close (цены закрытия).
# - **fast, slow, signal** — периоды MACD (по умолчанию 12, 26, 9).
#
# Действие:
# - Вычисляет MACD, сигнальную линию и гистограмму с помощью pandas_ta.
# - Добавляет бинарный признак **macd_long_signal**:\
# 1 — если MACD выше сигнальной линии (лонговый режим).\
# 0 — если MACD ниже или равен сигнальной линии.
#
# Выход:\
# Исходный df с новым признаком:\
# **macd_long_signal** (бинарный признак).

# %%
def macd_cross(df, fast=12, slow=26, signal=9):
    macd_data = ta.macd(df['Close'], fast=fast, slow=slow, signal=signal)
    df['macd_long_signal'] = (macd_data['MACD_12_26_9'] > macd_data['MACDs_12_26_9']).astype(int)
    return df


# %% [markdown]
# ## delta_macd

# %% [markdown]
# 📌 Рассчитывает относительные изменения MACD-компонент от текущей свечи\
# и добавляет признак наличия лонгового пересечения за последние cross_lookback свечей.
#
# Параметры:
# - fast, slow, signal — стандартные параметры MACD
# - shifts — список отступов (в свечах) от текущей для расчёта относительного изменения
# - cross_lookback — кол-во последних свечей, в которых ищется лонговое пересечение
#

# %%
def delta_macd(df, fast=12, slow=26, signal=9, shifts=[1, 5, 10, 20], cross_lookback=5):
   

    df = df.copy()

    # Считаем MACD и его компоненты во временных переменных
    macd_df = ta.macd(df['Close'], fast=fast, slow=slow, signal=signal)
    macd = macd_df[f'MACD_{fast}_{slow}_{signal}']
    macd_signal = macd_df[f'MACDs_{fast}_{slow}_{signal}']
    macd_hist = macd_df[f'MACDh_{fast}_{slow}_{signal}']

    # Относительные изменения от текущей свечи
    for name, series in [('macd', macd), ('macd_signal', macd_signal), ('macd_hist', macd_hist)]:
        for shift in shifts:
            shifted_vals = series.shift(shift)
            df[f'{name}_rel_change_{shift}'] = (series - shifted_vals) / (shifted_vals + 1e-9)

    # Лонговое пересечение MACD за последние N свечей
    macd_cross_long = (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
    df['macd_long_signal_lastN'] = macd_cross_long.rolling(cross_lookback, min_periods=1).max().astype(int)

    return df


# %% [markdown]
# # ATR

# %% [markdown]
# ## delta_atr

# %% [markdown]
# 📌 Описание функции **delta_atr** *(df, atr_length=14, window_size=20, num_blocks=4)*  
# **Вход:**
# - **df** — DataFrame с историческими свечами и обязательными колонками High, Low, Close.
# - **atr_length** — период ATR (например, 14).
# - **window_size** — общее кол-во свечей для анализа.
# - **num_blocks** — на сколько частей делим окно.
#
# **Что делает:**  
# Для заданного **atr_length**:\
# Считает ATR через `pandas_ta.atr()`.\
# Делит окно на **num_blocks** равных частей и для каждого блока считает относительное изменение ATR:
# $$\frac{ATR_{end} - ATR_{start}}{ATR_{start}}$$
#
# Определяет, в скольких блоках ATR рос (значения в диапазоне 0–num_blocks при стандартных настройках).
#
# **Выход:**  
# Возвращает тот же df, но с добавленными колонками:  
# `block1_atrXX_rel_change`, ..., `blockN_atrXX_rel_change` — относительные изменения ATR для каждого блока.  
# `atrXX_grow_blocks` — количество растущих блоков для данного ATR.

# %%
def delta_atr(df, atr_length=14, window_size=20, num_blocks=4):
    eps = 1e-9
    
    block_size = window_size // num_blocks
    shifts = [block_size * i for i in range(1, num_blocks + 1)]

    df = df.copy()
    
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=atr_length)
    atr = atr / df['High'].rolling(window_size).max()  # нормализация

    for i, shift in enumerate(shifts, 1):
        start_vals = atr.shift(shift - block_size)
        end_vals = atr.shift(shift)
        df[f'block{i}_atr{atr_length}_rel_change'] = (end_vals - start_vals) / (start_vals + eps)

    grow_flags = [(atr.shift(s) > atr.shift(s - block_size)) for s in shifts]
    df[f'atr{atr_length}_grow_blocks'] = pd.concat(grow_flags, axis=1).sum(axis=1).astype(int)

    return df


# %% [markdown]
# # Volume

# %% [markdown]
# ## delta_ema_volume

# %%
def delta_ema_volume(df, ema_lengths=[20], window_size=20, num_blocks=4):
    volume_norn = df['Volume'] / df['Volume'].rolling(20).max()
    block_size = window_size // num_blocks
    shifts = [block_size * i for i in range(1, num_blocks + 1)]

    df = df.copy()

    for ema_length in ema_lengths:
        ema = ta.ema(volume_norn, ema_length)

        # Изменения EMA в каждом блоке
        for i, shift in enumerate(shifts, 1):
            start_vals = ema.shift(shift - block_size)
            end_vals = ema.shift(shift)
            df[f'block{i}_ema_volume{ema_length}_rel_change'] = (end_vals - start_vals) / (start_vals + eps)

        # Кол-во растущих блоков
        grow_flags = [(ema.shift(s) > ema.shift(s - block_size)) for s in shifts]
        df[f'ema_volume{ema_length}_grow_blocks'] = pd.concat(grow_flags, axis=1).sum(axis=1).astype(int)

    return df


# %% [markdown]
# # Линейная регрессия

# %% [markdown]
# 📌 Описание функции **regression_slope_price** *(df, n=[5, 10, 20, 60])*  \
# Построение угла наклона линейной регрессии на окне n для цены

# %%
def regression_slope_price(df, n=[5, 10, 20, 60]):
    df = df.copy()
    close_norn = df['Close'] / df['Close'].rolling(20).max()
    for window in n:
        df[f'price_slope_{window}'] = ta.slope(close_norn, length=window)        
    return df


# %% [markdown]
# 📌 Описание функции **regression_slope_volume** *(df, n=[5, 10, 20, 60])*  \
# Построение угла наклона линейной регрессии на окне n для объема

# %%
def regression_slope_volume(df, n=[5, 10, 20, 60]):
    volume_norn = df['Volume'] / df['Volume'].rolling(20).max()
    df = df.copy()
    for window in n:
        df[f'volume_slope_{window}'] = ta.slope(volume_norn, length=window)
    return df


# %% [markdown]
# # RSI

# %% [markdown]
# ## delta_rsi

# %% [markdown]
# 📌 Аналог delta_ema, но для RSI.\
# Помимо относительного изменения в каждом блоке считает\
# среднее значение RSI внутри блока.
#
# Параметры:
# - df — DataFrame с колонкой Close
# - rsi_lengths — список длин RSI
# - window_size — общее кол-во свечей для анализа
# - num_blocks — на сколько частей делим окно
#

# %%
def delta_rsi(df, rsi_lengths=[21], window_size=20, num_blocks=4):
  
    close_norm = df['Close'] / df['Close'].rolling(window_size).max()
    block_size = window_size // num_blocks
    shifts = [block_size * i for i in range(1, num_blocks + 1)]

    df = df.copy()

    for rsi_length in rsi_lengths:
        rsi = ta.rsi(close_norm, length=rsi_length)

        # Изменения RSI и среднее значение по каждому блоку
        for i, shift in enumerate(shifts, 1):
            start_vals = rsi.shift(shift - block_size)
            end_vals = rsi.shift(shift)
            df[f'block{i}_rsi{rsi_length}_rel_change'] = (end_vals - start_vals) / start_vals

            block_vals = rsi.shift(shift - block_size).rolling(block_size).mean()
            df[f'block{i}_rsi{rsi_length}_mean'] = block_vals

        # Кол-во растущих блоков
        grow_flags = [(rsi.shift(s) > rsi.shift(s - block_size)) for s in shifts]
        df[f'rsi{rsi_length}_grow_blocks'] = pd.concat(grow_flags, axis=1).sum(axis=1).astype(int)

    return df


# %% [markdown]
# ## rsi_speed

# %% [markdown]
# Добавляет признаки скорости и ускорения RSI.
# Входные данные:
# - Периоды RSI для расчета (по умолчанию [21])
# - Окна для расчета скорости/ускорения (по умолчанию [10, 30, 60])
#
# На выходе получаем первую и вторую производную RSI

# %%
def rsi_speed(df, lengths=[21], windows=[10, 30, 60]):

    df = df.copy()
    
    for length in lengths:
        # Рассчитываем RSI (предварительно нормируем от 0 до 1)
        rsi_col = ta.rsi(df['Close'], length=length) / 100  # Нормировка
        
        for window in windows:
            # Скорость = среднее изменение RSI за window свечей
            speed_col = f'rsi{length}_speed_{window}'
            df[speed_col] = (rsi_col - rsi_col.shift(window)) / window
            
            # Ускорение = изменение скорости
            accel_col = f'rsi{length}_accel_{window}'
            df[accel_col] = df[speed_col].diff(window) / window
            
    return df

# %%
