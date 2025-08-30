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
# # Модуль буфера свечей для мониторинга монет

# %% [markdown]
# # Импорт библиотек 

# %%
# Модуль буфера свечей для мониторинга монет
import sys
import time
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from threading import Thread, Lock
import requests
sys.path.append("C:\\Users\\nikita\\Documents\\MagisterML\\MyJupiter\\trading_project\\")
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pybit.unified_trading import HTTP
import numpy as np
import contextlib
import io

from modules.indicators import delta_ema, macd_cross, delta_atr, delta_ema_volume, macd, regression_slope_price, regression_slope_volume, delta_rsi, ema_above_price, ema_speed, rsi_speed, rsi_divergence,\
    tema_slope_change, ema_price_distance, apply_indicators
from modules.create_df import fetch_kline_data, kline_candles
from monitoring import api_info, telegram_info


# %% [markdown]
# # Глобальные переменные

# %%
# Глобальные переменные
MODEL = None
THRESHOLD = None
FEATURE_COLUMNS = None
RUNNING = True

candle_buffers = defaultdict(dict)
buffer_locks = defaultdict(Lock)
last_update_time = {}

SYMBOLS = ['LINKUSDT', 'BCHUSDT']
BUFFER_SIZE = 300
MAX_WORKERS = min(10, len(SYMBOLS)) #количество параллельного мониторинга символов
TF_MINUTES = 1
MONITOR_INTERVAL = 1 * 60  # как часто обновлять данные (в секундах) 15 * 60 для м15

api_key = api_info.api_key
api_secret = api_info.api_secret

TELEGRAM_CHAT_ID = telegram_info.TELEGRAM_CHAT_ID
TELEGRAM_BOT_TOKEN = telegram_info.TELEGRAM_BOT_TOKEN

BYBIT_CLIENT = None  # глобальная переменная


# %% [markdown]
# # Загрузка модели

# %% [markdown]
# ## set_model_and_params

# %%
def set_model_and_params(model, threshold, feature_columns):
    """
    Устанавливает модель и параметры для мониторинга.
    """
    global MODEL, THRESHOLD, FEATURE_COLUMNS
    MODEL = model
    THRESHOLD = threshold
    FEATURE_COLUMNS = feature_columns


# %% [markdown]
# # Начало мониторинга

# %% [markdown]
# ## start_monitoring

# %%
def start_monitoring(model, threshold, feature_columns, 
                     send_telegram=True, send_orders=False):
    """
    Запуск мониторинга с моделью.
    """
    global MODEL, THRESHOLD, FEATURE_COLUMNS, SEND_TELEGRAM, SEND_ORDERS, RUNNING
    
    MODEL = model
    THRESHOLD = threshold
    FEATURE_COLUMNS = feature_columns
    SEND_TELEGRAM = send_telegram
    SEND_ORDERS = send_orders
    RUNNING = True

    if SEND_ORDERS:
        if not init_bybit_client(api_key, api_secret):
            print("⚠️ Не удалось подключиться к Bybit, торговля отключена")
            SEND_ORDERS = False

    print("🚀 Инициализация всех монет...")
    for symbol in SYMBOLS:
        try:
            init_buffer(symbol)
            print(f"✅ Инициализировано {symbol}, {BUFFER_SIZE} свечей")
        except Exception as e:
            print(f"⚠️ Пропущено {symbol} — ошибка: {e}")

    print("Запуск мониторинга...")
    monitor_thread = Thread(target=monitor_loop, daemon=False)
    monitor_thread.start()


# %% [markdown]
# ## stop_monitoring

# %%
def stop_monitoring():
    """Корректная остановка мониторинга"""
    global RUNNING
    RUNNING = False
    print("🛑 Остановка мониторинга...")


# %% [markdown]
# # Подключение к бирже для торговли

# %% [markdown]
# ## init_bybit_client

# %%
def init_bybit_client(api_key, api_secret, testnet=False):
    """Инициализация клиента Bybit API"""
    global BYBIT_CLIENT
    try:
        BYBIT_CLIENT = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            recv_window=15000  # увеличенное окно для избежания ошибок времени
        )
        print("✅ Bybit клиент успешно инициализирован")
        return True
    except Exception as e:
        print(f"❌ Ошибка инициализации Bybit клиента: {e}")
        return False


# %% [markdown]
# # Инициализация первичного буфера свечей

# %% [markdown]
# ## init_buffer

# %%
def init_buffer(symbol):
    end = int(time.time() * 1000)
    start = end - BUFFER_SIZE * TF_MINUTES * 60 * 1000

    raw_df = fetch_kline_data(symbol, str(TF_MINUTES), start, end)

    if raw_df is None or raw_df.empty:
        print(f"⚠️ Не удалось получить данные для инициализации {symbol}")
        return

    # Шаг 1: обработка индикаторами
    processed_df = apply_indicators(raw_df.copy())

    # Шаг 2: оставляем только признаки для модели
    processed_df = processed_df[FEATURE_COLUMNS]
    processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]

    # Шаг 3: latest_row для предсказаний
    latest_row = processed_df.iloc[-1].copy()

    # Шаг 4: сохраняем в буфер
    with buffer_locks[symbol]:
        candle_buffers[symbol] = {
            'raw_data': raw_df.tail(BUFFER_SIZE).reset_index(drop=True),   # OHLCV
            'processed_data': processed_df.copy(),                        # признаки для модели
            'latest_row': latest_row
        }

    #print(f"✅ Буфер инициализирован: {symbol}, {len(processed_df)} строк")
    # if symbol == 'LINKUSDT':
    #     print("📈 Последний бар (OHLCV):")
    #     print(raw_df.tail(1).to_string(index=False))


# %% [markdown]
# # Обновление буфера

# %% [markdown]
# ## update_buffer

# %%
def update_buffer(symbol):
    """Обновляет буфер данных с последней свечой"""
    try:
        if MODEL is None or FEATURE_COLUMNS is None:
            raise RuntimeError("Модель или признаки не инициализированы")

        new_candle = fetch_latest_candle(symbol, TF_MINUTES)
        if new_candle is None or new_candle.empty:
            print(f"⚠️ Нет новой свечи для {symbol}")
            return "retry"

        with buffer_locks[symbol]:
            buffer = candle_buffers.get(symbol)
            if not buffer or buffer['raw_data'].empty:
                print(f"⚠️ Буфер пуст для {symbol}, перезапускаем инициализацию")
                init_buffer(symbol)
                return "retry"

            last_time = buffer['raw_data']["Date"].iloc[-1]
            new_time = new_candle["Date"].iloc[-1]

            if new_time <= last_time:
                print(f"⏭ Данные не обновились для {symbol}")
                return "skip"

            # Объединяем новые свечи с буфером
            full_df = pd.concat([buffer['raw_data'], new_candle.tail(1)], ignore_index=True).tail(BUFFER_SIZE)

            # Обработка индикаторами
            processed_df = apply_indicators(full_df.copy())
            processed_df = processed_df[FEATURE_COLUMNS]
            processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]

            if processed_df.empty:
                print(f"⚠️ После обработки свеча {symbol} не содержит данных. Возможно, недостаточно данных для индикаторов.")
                return "retry"

            missing = set(FEATURE_COLUMNS) - set(processed_df.columns)
            if missing:
                raise ValueError(f"Отсутствуют фичи: {missing}")

            latest_row = processed_df.iloc[-1].copy()

            # Обновляем буфер
            candle_buffers[symbol] = {
                'raw_data': full_df,                        # OHLCV
                'processed_data': processed_df.copy(),     # признаки для модели
                'latest_row': latest_row
            }

            # print(f"🔄 Обновлено: {symbol} до {new_time}")
            if symbol == 'LINKUSDT':
                print(full_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(1).to_string(index=False))

            return "ok"

    except Exception as e:
        print(f"❌ Ошибка при обновлении {symbol}: {str(e)[:200]}")
        return "retry"


# %% [markdown]
# # Мониторинг

# %% [markdown]
# ## Мониторинг

# %% [markdown]
# ### monitor_loop

# %%
def monitor_loop(send_telegram=True, send_orders=False):
    """Основной цикл мониторинга"""
    global TIME_DIFF, RUNNING

    sync_bybit_time_simple()
    original_time = time.time
    time.time = lambda: original_time() + TIME_DIFF / 1000

    print(f"🚀 Запуск мониторинга (TF={TF_MINUTES} мин, интервал={MONITOR_INTERVAL} сек)")

    try:
        wait_until_next_candle()

        while RUNNING:
            start = time.time()
            print(f"\n🔄 Цикл {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # --- Обновление буферов ---
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                update_results = list(executor.map(update_buffer, SYMBOLS))

            # Ручной повтор для "retry"
            for symbol, status in zip(SYMBOLS, update_results):
                if status == "retry":
                    print(f"🔁 Повтор обновления {symbol}")
                    update_buffer(symbol)

            # --- Предсказания и сигналы ---
            def process_and_alert(symbol):
                result = process_symbol(symbol, verbose=False)  # Убрали verbose вывод
                if result is not None:
                    latest_row, y_pred, y_proba, raw_latest = result
                    order_sent = send_alert(
                        symbol,
                        latest_row=latest_row,
                        raw_row=raw_latest,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        send_telegram=send_telegram,
                        send_orders=send_orders
                    )
                    return 1, order_sent
                return 0, False

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = list(executor.map(process_and_alert, SYMBOLS))
            
            signal_count = sum(result[0] for result in results)
            order_count = sum(result[1] for result in results)

            # --- Проверка рассинхронизации времени ---
            drift = check_bybit_server_time()
            if drift and drift > 2.0:
                print("🔁 Попытка синхронизации времени...")
                resync_system_time()
                time.sleep(2)
                sync_bybit_time_simple()
                time.time = lambda: original_time() + TIME_DIFF / 1000

            elapsed = time.time() - start
            print(
                f"\n📊 Цикл завершён:"
                f"\n- Обновлено: {update_results.count('ok')}/{len(SYMBOLS)}"
                f"\n- Сигналы: {signal_count}"
                f"\n- Ордера: {order_count}"
                f"\n- Время: {elapsed:.2f} сек"
            )

            if RUNNING:
                time.sleep(max(0, MONITOR_INTERVAL - elapsed))

    except KeyboardInterrupt:
        print("\n🛑 Мониторинг остановлен вручную")
    except Exception as e:
        print(f"❌ Ошибка в цикле: {str(e)[:200]}")
    finally:
        print("✅ Мониторинг завершён")


# %% [markdown]
# ### wait_until_next_candle

# %%
def wait_until_next_candle(tf_minutes=1):
    """Ждёт до конца текущей свечи TF."""
    now = datetime.utcnow()  # UTC без tz
    total_seconds = now.minute * 60 + now.second
    seconds_to_wait = tf_minutes * 60 - (total_seconds % (tf_minutes * 60))
    print(f"⏳ Ждём {seconds_to_wait} секунд до закрытия свечи...")
    time.sleep(seconds_to_wait + 1)


# %% [markdown]
# ## Синхронизация времени

# %% [markdown]
# ### sync_bybit_time_simple

# %%
def sync_bybit_time_simple():
    """Синхронизация через API Bybit без системных команд"""
    global TIME_DIFF
    try:
        server_time = int(requests.get(
            "https://api.bybit.com/v5/market/time",
            timeout=3
        ).json()["result"]["timeNano"]) // 1_000_000
        
        local_time = int(time.time() * 1000)
        TIME_DIFF = server_time - local_time
        print(f"⏱ Коррекция времени: {TIME_DIFF/1000:.3f} сек")
        
    except Exception as e:
        print(f"⚠️ Не удалось синхронизировать время: {e}")
        TIME_DIFF = 0


# %% [markdown]
# ### def check_bybit_server_time

# %%
def check_bybit_server_time():
    try:
        response = requests.get("https://api.bybit.com/v5/market/time", timeout=5)
        response.raise_for_status()
        data = response.json()
        server_time_ms = int(data["result"]["timeNano"]) // 1_000_000
        local_time_ms = int(time.time() * 1000)
        drift = abs(server_time_ms - local_time_ms) / 1000
        if drift > 3:
            print(f"⚠️ Расхождение с Bybit временем: {drift:.2f} сек")
        else:
            print(f"✅ Время синхронизировано с Bybit (расхождение: {drift:.2f} сек)")
        return drift
    except Exception as e:
        print(f"❌ Не удалось получить время Bybit: {e}")
        return None


# %% [markdown]
# ## Предсказания

# %% [markdown]
# ### process_symbol

# %%
def process_symbol(symbol, verbose=True):
    """Предсказание для одного символа"""
    if MODEL is None or FEATURE_COLUMNS is None:
        if verbose:
            print(f"[{symbol}] ❌ MODEL или FEATURE_COLUMNS не заданы!")
        return None

    try:
        with buffer_locks[symbol]:
            buffer = candle_buffers.get(symbol)
            if not buffer:
                if verbose:
                    print(f"❌ Буфер не найден для {symbol}")
                return None

            latest_row = buffer.get('latest_row')
            processed_data = buffer.get('processed_data')

        if latest_row is None or processed_data is None or processed_data.empty:
            if verbose:
                print(f"⚠️ Нет данных для {symbol}")
            return None

        missing = [col for col in FEATURE_COLUMNS if col not in processed_data.columns]
        if missing:
            print(f"❌ Отсутствующие фичи в {symbol}: {missing}")
            return None

        # Преобразование в массив для модели
        try:
            X = latest_row[FEATURE_COLUMNS].astype(float).values.reshape(1, -1)
        except Exception as e:
            print(f"❌ Преобразование признаков {symbol} не удалось: {e}")
            return None

        if not np.isfinite(X).all():
            if verbose:
                print(f"⚠️ Невалидные значения: {X}")
            return None

        # Предсказание
        y_proba = MODEL.predict_proba(X)[0, 1]
        y_pred = int(y_proba > THRESHOLD)

        # Убрал print отсюда - будет в send_alert
        if y_pred == 1:
            # Получаем сырые данные для алерта
            with buffer_locks[symbol]:
                buffer = candle_buffers.get(symbol)
                raw_latest = buffer['raw_data'].iloc[-1] if buffer else None
            
            return latest_row, y_pred, y_proba, raw_latest  # для monitor_loop

        return None

    except Exception as e:
        print(f"❌ Ошибка в {symbol}: {str(e)[:150]}...")
        return None


# %% [markdown]
# ## Отправка уведомлений

# %% [markdown]
# ### send_alert

# %%
def send_alert(symbol, latest_row, raw_row, y_pred=None, y_proba=None, send_telegram=True, send_orders=False):
    """Централизованная функция для обработки и вывода сигналов"""
    try:
        candle_time = pd.to_datetime(raw_row['Date']).strftime("%Y-%m-%d %H:%M:%S")
        price = float(raw_row['Close'])
        alert_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Если y_pred/y_proba не переданы, делаем предсказание
        if y_pred is None or y_proba is None:
            X = latest_row[FEATURE_COLUMNS].values.reshape(1, -1)
            y_proba = MODEL.predict_proba(X)[0, 1]
            y_pred = int(y_proba > THRESHOLD)

        # --- ЦЕНТРАЛИЗОВАННЫЙ ВЫВОД В КОНСОЛЬ ---
        if y_pred == 1:
            print(f"\n🚨 СИГНАЛ: {symbol} | Время свечи: {candle_time} | "
                  f"Цена: {price:.6f} | Вероятность: {y_proba:.4f} | "
                  f"Порог: {THRESHOLD:.2f} → Предсказание: {y_pred}")
        else:
            print(f"📉 {symbol}: proba={y_proba:.4f}, threshold={THRESHOLD:.2f} → pred={y_pred}")

        # Сообщение для Telegram
        message = (
            f"🚨 **Торговый сигнал**\n"
            f"**Монета:** {symbol}\n"
            f"**Время свечи:** {candle_time} (UTC)\n"
            f"**Цена закрытия:** {price:.6f}\n"
            f"**Вероятность:** {y_proba:.4f}\n"
            f"**Предсказание:** {y_pred}\n"
        )

        # Торговое решение
        order_sent = False
        if send_orders and y_pred == 1:
            order_sent = place_buy_order_with_checks(symbol, price)
            order_message = "✅ Ордер отправлен" if order_sent else "⚠️ Условия для ордера не выполнены"
            message += f"\n\n{order_message}"

        # Отправка Telegram
        if send_telegram and y_pred == 1:  # Отправляем только сигналы
            send_telegram_alert(message)

        return order_sent

    except Exception as e:
        print(f"❌ Ошибка в send_alert для {symbol}: {e}")
        return False


# %% [markdown]
# ### send_telegram_alert

# %%
def send_telegram_alert(message):
    """
    Универсальная отправка сообщения в Telegram.
    """
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()

        if not data.get("ok"):
            print(f"⚠️ Telegram API вернул ошибку: {data}")
    except Exception as e:
        print(f"❌ Ошибка отправки в Telegram: {e}")


# %% [markdown]
# # Загрузка одной свечи с bybit

# %% [markdown]
# ## fetch_latest_candle

# %%
def fetch_latest_candle(symbol, tf):
    """Получает последнюю ЗАКРЫТУЮ свечу для символа"""
    url = "https://api.bybit.com/v5/market/kline"
    
    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': str(int(tf)),
        'limit': 2  # Запрашиваем 2 свечи
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        result = r.json().get('result', {})
        candles = result.get('list', [])
        
        if not candles or len(candles) < 2:
            print(f"⚠️ Недостаточно данных для {symbol}")
            return None

        # Берем ПРЕДЫДУЩУЮ свечу (вторая в списке), которая гарантированно закрыта
        closed_candle = candles[1]  # ← Это важно!

        df = pd.DataFrame([closed_candle], columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

        df["Date"] = pd.to_datetime(df["Date"].astype(float), unit="ms")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)

        print(f"✅ Получена закрытая свеча {symbol}: {df['Date'].iloc[0]} - Close: {df['Close'].iloc[0]}")
        return df

    except Exception as e:
        print(f"❌ Ошибка при получении свечи {symbol}: {e}")
        return None

# %%
