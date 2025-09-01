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
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm
import re
import joblib
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import time
from datetime import timedelta, datetime
import os
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, roc_auc_score, precision_recall_curve, precision_recall_curve, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
import shap
from ipywidgets import widgets
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import display


# %% [markdown]
# # Разметка данных

# %% [markdown]
# ## plot_ohlc_signals

# %% [markdown]
# Рисует график OHLC с подсветкой buy/sell сигналов и шумных сигналов
#     
# Аргументы:
# - df с колонками *'Open','High','Low','Close','buy','sell','buy_noised','sell_noised'*
# - **start_idx**: начальный индекс участка
# - **end_idx**: конечный индекс участка

# %%
def plot_ohlc_signals(df, start_idx=0, end_idx=None):
    """
    Рисует график OHLC с подсветкой buy/sell сигналов и шумных сигналов
    
    Args:
        df: DataFrame с колонками 'Open','High','Low','Close','buy','sell','buy_noised','sell_noised'
        start_idx: начальный индекс участка
        end_idx: конечный индекс участка
    """
    if end_idx is None:
        end_idx = len(df)
    
    plot_data = df.iloc[start_idx:end_idx].copy()
    
    plt.figure(figsize=(16, 8))
    
    # Рисуем все ценовые линии
    plt.plot(plot_data.index, plot_data['Close'], 'b-', label='Close', linewidth=1.5)
    plt.plot(plot_data.index, plot_data['Open'], 'g--', label='Open', linewidth=1, alpha=0.7)
    plt.plot(plot_data.index, plot_data['High'], 'c:', label='High', linewidth=1, alpha=0.7)
    plt.plot(plot_data.index, plot_data['Low'], 'm:', label='Low', linewidth=1, alpha=0.7)
    
   
     # Шумные сигналы покупки (более прозрачные и меньшего размера)
    buy_noised_signals = plot_data[plot_data['buy_noised'] == 1]
    if not buy_noised_signals.empty:
        plt.scatter(buy_noised_signals.index, buy_noised_signals['Close'], 
                   color='blue', marker='^', s=80, label='Buy noised', 
                   zorder=3, alpha=0.6, edgecolors='darkgreen', linewidth=0.5)
        
    # Основные сигналы покупки
    buy_main_signals = plot_data[plot_data['buy'] == 1]
    if not buy_main_signals.empty:
        plt.scatter(buy_main_signals.index, buy_main_signals['Close'], 
                   color='lightgreen', marker='^', s=80, label='Buy main', 
                   zorder=3, alpha=0.6, edgecolors='darkgreen', linewidth=0.5)
    
    # Основные сигналы продажи
    sell_noised_signals = plot_data[plot_data['sell'] == 1]
    if not sell_noised_signals.empty:
        plt.scatter(sell_noised_signals.index, sell_noised_signals['Close'], 
                   color='lightcoral', marker='v', s=80, label='Sell main', 
                   zorder=3, alpha=0.6, edgecolors='darkred', linewidth=0.5)
    
    # Финальные сигналы покупки (более яркие и крупные)
    buy_signals = plot_data[plot_data['buy_strong'] == 1]
    if not buy_signals.empty:
        plt.scatter(buy_signals.index, buy_signals['Close'], 
                   color='green', marker='^', s=20, label='Buy Strong', zorder=5)
    
    # Финальные сигналы продажи (более яркие и крупные)
    sell_signals = plot_data[plot_data['sell_strong'] == 1]
    if not sell_signals.empty:
        plt.scatter(sell_signals.index, sell_signals['Close'], 
                   color='red', marker='v', s=20, label='Sell Strong', zorder=5)
    
    plt.title('OHLC Prices with Buy/Sell Signals')
    plt.xlabel('Candles')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# # Корреляционный анализ

# %% [markdown]
# ## plot_corr_by_distance

# %% [markdown]
# **plot_corr_by_distance** Строит график средней корреляции (по модулю) между target и всеми признаками.
# - по оси **y** указаны значения корреляции между целевой переменной **target** и признаками
# - по оси **x** отложено расстояние от текущей закрытой свечи до свечей на которых был проведен расчет признака

# %%
def plot_corr_by_distance(df, target='buy', pattern=r'_(\d+)$', min_corr=0.01):
    """
    Строит график средней корреляции (по модулю) между target и признаками,
    где в названии признака есть число (удаление от текущей цены).
    В расчет берутся только признаки с |corr| > min_corr.
    """
    corr = df.corr(numeric_only=True)[target].drop(target).abs()

    # Фильтрация слабых корреляций
    corr = corr[corr > min_corr]

    distance_corrs = {}
    for col, val in corr.items():
        match = re.search(pattern, col)
        if match:
            dist = int(match.group(1))
            distance_corrs.setdefault(dist, []).append(val)

    avg_corr = {dist: np.mean(vals) for dist, vals in distance_corrs.items()}

    distances = sorted(avg_corr.keys())
    values = [avg_corr[d] for d in distances]

    plt.figure(figsize=(8, 4))
    plt.plot(distances, values, marker='o')
    plt.title('График угасания корреляции')
    plt.xlabel('Расстояние от текущей свечи')
    plt.ylabel('Средняя |corr|')
    plt.grid(True)
    plt.show()


# %% [markdown]
# ## plot_correlation_matrix

# %% [markdown]
# **plot_correlation_matrix** Строит тепловую карту корреляций целевой переменной и признаками
#
# Вход:
# - df с признаками и целевой переменной
# - drop_columns базовые/промежуточные колонки не требующие исследования
# - top_n - фильтр по корреляции
# - target - целевая переменная

# %%
def plot_correlation_matrix(df, target=None, drop_columns=['Data', 'High', 'Low', 'Close', 'Open', 'Volume'], top_n=30):
    """
    Строит корреляционную матрицу для топ-N признаков, наиболее коррелированных с целевой переменной
    
    Args:
        df: DataFrame с признаками
        target: имя целевой переменной (если None - корреляция между всеми признаками)
        drop_columns: колонки для исключения из анализа
        top_n: количество признаков для отображения
    """
    try:
        # Удаляем ненужные колонки
        data = df.drop(drop_columns, axis=1, errors='ignore')
        
        # Если указана целевая переменная - выбираем топ-N признаков по корреляции с ней
        if target is not None and target in data.columns:
            # Вычисляем корреляцию с целевой переменной
            target_corr = data.corr()[target].abs().sort_values(ascending=False)
            
            # Берем топ-N признаков (включая саму целевую)
            top_features = target_corr.head(top_n).index.tolist()
            
            # Убедимся, что целевая переменная есть в списке
            if target not in top_features:
                top_features.append(target)
                
            corr_matrix = data[top_features].corr()
            
        else:
            # Стандартный подход - корреляция между всеми признаками
            corr_matrix = data.corr()
            
            # Если признаков слишком много - ограничиваем топ-N
            if len(corr_matrix) > top_n:
                mean_abs_corr = corr_matrix.abs().mean().sort_values(ascending=False)
                top_features = mean_abs_corr.head(top_n).index
                corr_matrix = corr_matrix.loc[top_features, top_features]
        
        num_features = len(corr_matrix)
        
        # Динамические настройки в зависимости от количества признаков
        if num_features <= 15:
            figsize = (10, 8)
            font_scale = 1.2
            annot = True
            label_size = 10
        elif num_features <= 30:
            figsize = (16, 14)
            font_scale = 1.0
            annot = False
            label_size = 9
        else:
            figsize = (20, 18)
            font_scale = 0.8
            annot = False
            label_size = 8
            plt.rcParams['xtick.major.pad'] = 0.5
            plt.rcParams['ytick.major.pad'] = 0.5
        
        # Настройка стиля
        sns.set(font_scale=font_scale)
        plt.figure(figsize=figsize)
        
        # Построение тепловой карты
        heatmap = sns.heatmap(
            corr_matrix,
            cmap='coolwarm',
            annot=annot,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.7},
            mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
            annot_kws={"size": 8} if annot else None
        )
        
        # Настройка подписей осей
        heatmap.set_xticklabels(
            heatmap.get_xticklabels(),
            rotation=45,
            ha='right',
            fontsize=label_size
        )
        heatmap.set_yticklabels(
            heatmap.get_yticklabels(),
            rotation=0,
            fontsize=label_size
        )
        
        # Формируем заголовок
        if target is not None and target in data.columns:
            title = f'Корреляционная матрица (топ-{num_features} признаков с "{target}")'
        else:
            title_suffix = f' (топ-{num_features} из {len(data.columns)} признаков)' if len(data.columns) > num_features else f' ({num_features} признаков)'
            title = f'Корреляционная матрица{title_suffix}'
            
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Ошибка при построении графика: {str(e)}")


# %% [markdown]
# # Визуализация признаков

# %% [markdown]
# **plot_price_with_indicators** Строит основной график цены и визуализирует используемые признаки - индикаторы.
#
# На вход подается:
# - df с колонкой цены **Close**
# - исследуемые границы графика - **start / end**
# - заголовок графика
# - список индикаторов **indicators**

# %%
def plot_price_with_indicators(df, indicators, start=-400, end=-200, colors=None, title=None):
    """
    df          - DataFrame с колонками Close и индикаторами
    indicators  - список названий индикаторов для отображения
    start, end  - диапазон среза df
    colors      - список цветов (опционально)
    title       - заголовок графика (опционально)
    """
    df_slice = df.iloc[start:end]
    
    # Настройка стиля графика
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6), facecolor='#f8f9fa')
    
    # цена - основная линия (жирная и четкая)
    plt.plot(df_slice['Close'], label='Close', color='#2c3e50', linewidth=2.5, alpha=0.9)
    plt.ylabel('Close Price', fontsize=12)
    plt.xlabel('Minutes', fontsize=12)
    
    # Настройка фона области графика
    ax = plt.gca()
    ax.set_facecolor('#f0f3f5')
    
    # индикаторы на втором axes
    ax2 = plt.twinx()
    ax2.set_ylabel('Indicators', fontsize=12)
    ax2.set_facecolor('#f0f3f5')
    
    if colors is None:
        # Приглушенные цвета для индикаторов
        colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#8e44ad', 
                 '#16a085', '#d35400', '#2c3e50', '#7f8c8d', '#9b59b6']
    
    # Элегантные стили линий
    line_styles = ['--', '-.', ':', '--', '-.', ':']
    
    for i, ind in enumerate(indicators):
        # Штриховые линии с хорошей прозрачностью
        ax2.plot(df_slice[ind], label=ind, 
                color=colors[i % len(colors)], 
                linestyle=line_styles[i % len(line_styles)],
                linewidth=1.8, 
                alpha=0.7)  # оптимальная прозрачность
    
    # легенда
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
              frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # Настройка сетки
    ax.grid(True, linestyle='--', alpha=0.3)  # очень легкая сетка
    ax2.grid(False)
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()


# %% [markdown]
# # Расчёт и визуализация важности признаков на основе Mutual Information (до моделей)

# %% [markdown]
# **mutual_info_classif** вычисляет взаимную информацию между каждым признаком и целевой переменной.\
#
# Входные данные:
# - X_train — матрица признаков
# - y_train — целевая переменная
# - top_n — количество топ-признаков для отображения
# - random_state — seed для воспроизводимости
#
# Процесс работы:
# - Вычисляет Mutual Information между каждым признаком и целевой переменной
# - Сортирует признаки по убыванию важности
# - Выводит таблицу топ-N наиболее информативных признаков
# - Строит горизонтальный барчарт для наглядной визуализации
#
# Особенности:
# - Работает с категориальными и числовыми признаками
# - Оценивает нелинейные зависимости: Может выявить сложные связи, которые пропускает линейная корреляция

# %%
def explain_model_mutual_info(X_train, y_train, top_n=20, random_state=3):
    """
    Расчёт важности признаков на основе Mutual Information.
    """
    try:
        start_time = time.time()
        print(f"ℹ️ Calculating Mutual Information for {X_train.shape[1]} features...")

        # 1. Расчёт MI
        mi_scores = mutual_info_classif(X_train, y_train, random_state=random_state)
        mi_df = pd.DataFrame({
            'Feature': X_train.columns,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)

        elapsed_time = time.time() - start_time
        print(f"✅ MI calculation completed in {elapsed_time:.2f} seconds")

        # 2. Таблица топ-N
        print(f"\n🔍 Top {top_n} Features by Mutual Information:")
        print(mi_df.head(top_n).to_markdown(index=False, floatfmt=".4f"))

        # 3. Визуализация
        plt.figure(figsize=(10, min(6, top_n * 0.3)))
        plt.barh(mi_df['Feature'].head(top_n)[::-1], 
                 mi_df['MI_Score'].head(top_n)[::-1], 
                 color='skyblue')
        plt.xlabel('Mutual Information Score')
        plt.title(f'Top {top_n} Features by Mutual Information')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Ошибка при расчёте Mutual Information: {str(e)}")


# %% [markdown]
# # Распределение целевой переменной внутри выборок

# %% [markdown]
# ## show_class_balance

# %% [markdown]
# **show_class_balance** Анализирует и визуализирует распределение классов по выборкам
#
# Вход:
# - y: целевая переменная всего датасета
# - y_train: обучающая выборка
# - y_valid: валидационная выборка
# - y_test: тестовая выборка
#
# Выход:
# - Таблица с долями классов в каждой выборке
# - Столбчатая диаграмма распределения
# - Визуальная проверка сбалансированности данных
#
# Что делает: Сравнивает пропорции классов между разными выборками для контроля репрезентативности разбиения

# %%
def show_class_balance(y, y_train, y_valid, y_test):
    # Собираем данные в таблицу
    balance_df = pd.DataFrame({
        'Весь датасет': y.value_counts(normalize=True).round(3),
        'Обучающая': y_train.value_counts(normalize=True).round(3),
        'Валидационная': y_valid.value_counts(normalize=True).round(3),
        'Тестовая': y_test.value_counts(normalize=True).round(3)
    }).fillna(0)  # на случай отсутствующих классов
    
    # Выводим таблицу в стиле "plain"
    print("📊 Баланс классов (доли):")
    print(
        balance_df.to_markdown(
            tablefmt="simple",  # Чистый формат без лишних линий
            stralign="center",  # Выравнивание по центру
            floatfmt=".3f"       # Формат чисел
        )
    )
    
    # Визуализация
    plt.figure(figsize=(10, 5))
    balance_df.plot(kind='bar', width=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title('Распределение классов по выборкам', pad=20)
    plt.ylim(0, 1)
    plt.ylabel('Доля класса')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(framealpha=0.9)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# # Анализ порога классификации

# %% [markdown]
# ## evaluate_model_with_threshold

# %% [markdown]
# **evaluate_model_with_threshold** Оценивает модель с подбором оптимального порога и возвращает результаты
#
# Вход:
# - model: обученная модель классификации
# - X_train, y_train: обучающая выборка
# - X_valid, y_valid: валидационная выборка
# - X_test, y_test: тестовая выборка (опционально)
#
# Выход:
# - Словарь с моделью, метриками и метаданными:
# - Обученная модель
# - Метрики (F1, Precision, Recall, ROC AUC) для всех выборок
# - Оптимальный порог классификации
# - Список использованных признаков

# %%
def evaluate_model_with_threshold(model, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None):
    """
    Оценивает модель и возвращает результаты в формате для сохранения
    
    Возвращает словарь в формате:
    {
        'model': model,  # обученная модель
        'metrics': {
            'train': {метрики},
            'valid': {метрики},
            'test': {метрики} (если есть),
            'optimal_threshold': float
        },
        'features': list,  # список фичей
        'timestamp': str   # время оценки
    }
    """
    from sklearn.metrics import roc_auc_score
    
    # 1. Получаем предсказанные вероятности
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_valid_proba = model.predict_proba(X_valid)[:, 1]
    
    if X_test is not None and y_test is not None:
        y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # 2. Создаем диапазон порогов
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # 3. Функция для вычисления F1 при разных порогах
    def find_best_threshold(y_true, y_proba, thresholds):
        f1_scores = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx], f1_scores
    
    # 4. Находим лучшие пороги для train и valid
    train_best_threshold, train_f1_scores = find_best_threshold(y_train, y_train_proba, thresholds)
    valid_best_threshold, valid_f1_scores = find_best_threshold(y_valid, y_valid_proba, thresholds)
    
    # 5. Вычисляем средний оптимальный порог
    optimal_threshold = np.mean([train_best_threshold, valid_best_threshold])
    
    # 6. Создаем словари с метриками
    train_metrics = {
        'thresholds': thresholds,
        'f1_scores': train_f1_scores,
        'precision': [precision_score(y_train, (y_train_proba >= t).astype(int), zero_division=0) for t in thresholds],
        'recall': [recall_score(y_train, (y_train_proba >= t).astype(int), zero_division=0) for t in thresholds],
        'y_proba': y_train_proba,
        'max_f1_threshold': train_best_threshold,
        'roc_auc': roc_auc_score(y_train, y_train_proba)  # Добавлено ROC AUC
    }
    
    valid_metrics = {
        'thresholds': thresholds,
        'f1_scores': valid_f1_scores,
        'precision': [precision_score(y_valid, (y_valid_proba >= t).astype(int), zero_division=0) for t in thresholds],
        'recall': [recall_score(y_valid, (y_valid_proba >= t).astype(int), zero_division=0) for t in thresholds],
        'y_proba': y_valid_proba,
        'max_f1_threshold': valid_best_threshold,
        'roc_auc': roc_auc_score(y_valid, y_valid_proba)  # Добавлено ROC AUC
    }
    
    # 7. Выводим результаты
    print(f"🎯 Лучший порог по F1 (Train): {train_best_threshold:.4f}")
    print(f"🎯 Лучший порог по F1 (Valid): {valid_best_threshold:.4f}")
    print(f"✅ Усредненный оптимальный порог: {optimal_threshold:.4f}")
    print(f"\n📊 ROC AUC Scores:")
    print(f"✅ Train ROC AUC: {train_metrics['roc_auc']:.4f}")
    print(f"✅ Valid ROC AUC: {valid_metrics['roc_auc']:.4f}")
    
    # 8. Считаем финальные метрики с усредненным порогом
    def calculate_final_metrics(y_true, y_proba, threshold, set_name):
        y_pred = (y_proba >= threshold).astype(int)
        metrics = {
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_true, y_proba)  # Добавлено ROC AUC
        }
        print(f"\n📊 {set_name} set (Threshold = {threshold:.4f}):")
        print(f"✅ F1: {metrics['F1']:.4f}")
        print(f"✅ Precision: {metrics['Precision']:.4f}")
        print(f"✅ Recall: {metrics['Recall']:.4f}")
        print(f"✅ ROC AUC: {metrics['ROC_AUC']:.4f}")
        return metrics
    
    train_metrics['final_metrics'] = calculate_final_metrics(y_train, y_train_proba, optimal_threshold, "Train")
    valid_metrics['final_metrics'] = calculate_final_metrics(y_valid, y_valid_proba, optimal_threshold, "Valid")
    
    results = {
        'train': train_metrics,
        'valid': valid_metrics,
        'optimal_threshold': optimal_threshold
    }
    
    if X_test is not None and y_test is not None:
        test_metrics = {
            'thresholds': thresholds,
            'f1_scores': [f1_score(y_test, (y_test_proba >= t).astype(int), zero_division=0) for t in thresholds],
            'precision': [precision_score(y_test, (y_test_proba >= t).astype(int), zero_division=0) for t in thresholds],
            'recall': [recall_score(y_test, (y_test_proba >= t).astype(int), zero_division=0) for t in thresholds],
            'y_proba': y_test_proba,
            'roc_auc': roc_auc_score(y_test, y_test_proba)  # Добавлено ROC AUC
        }
        test_metrics['final_metrics'] = calculate_final_metrics(
            y_test, y_test_proba, optimal_threshold, "Test"
        )
        results['test'] = test_metrics
    
    # 9. Визуализация (остается без изменений)
    plt.figure(figsize=(18, 6))
    
    # 1. Кривые для обучающей выборки
    plt.subplot(1, 3, 1)
    plt.plot(train_metrics['thresholds'], train_metrics['precision'], label='Precision', color='blue')
    plt.plot(train_metrics['thresholds'], train_metrics['recall'], label='Recall', color='green')
    plt.plot(train_metrics['thresholds'], train_metrics['f1_scores'], label='F1', color='red')
    plt.axvline(optimal_threshold, color='k', linestyle='-', label=f'Avg Optimal: {optimal_threshold:.3f}')
    plt.axvline(train_best_threshold, color='b', linestyle=':', label=f'Train Max F1: {train_best_threshold:.3f}')
    plt.title('Train Selection')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    
    # 2. Кривые для валидационной выборки
    plt.subplot(1, 3, 2)
    plt.plot(valid_metrics['thresholds'], valid_metrics['precision'], label='Precision', color='blue')
    plt.plot(valid_metrics['thresholds'], valid_metrics['recall'], label='Recall', color='green')
    plt.plot(valid_metrics['thresholds'], valid_metrics['f1_scores'], label='F1', color='red')
    plt.axvline(optimal_threshold, color='k', linestyle='-', label=f'Avg Optimal: {optimal_threshold:.3f}')
    plt.axvline(valid_best_threshold, color='orange', linestyle=':', label=f'Valid Max F1: {valid_best_threshold:.3f}')
    plt.title('Test Set')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    
    # 3. Сравнение F1-кривых с новым порогом
    plt.subplot(1, 3, 3)
    plt.plot(train_metrics['thresholds'], train_metrics['f1_scores'], label='Train F1', color='blue')
    plt.plot(valid_metrics['thresholds'], valid_metrics['f1_scores'], label='Valid F1', color='orange')
    
    # Добавлена третья линия для тестовой выборки, если она есть
    if X_test is not None and y_test is not None:
        plt.plot(test_metrics['thresholds'], test_metrics['f1_scores'], label='Test F1', color='green')
    
    plt.axvline(optimal_threshold, color='k', linestyle='-', label=f'Avg Optimal: {optimal_threshold:.3f}')
    plt.axvline(train_best_threshold, color='b', linestyle=':', alpha=0.5)
    plt.axvline(valid_best_threshold, color='orange', linestyle=':', alpha=0.5)
    plt.title('F1 Comparison with Optimal Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    # 10. Выводим итоговые метрики в таблице (добавляем ROC AUC)
    final_table = [
        ["Dataset", "Threshold Type"] + list(train_metrics['final_metrics'].keys()),
        ["Train", f"Average Optimal ({optimal_threshold:.4f})"] + list(train_metrics['final_metrics'].values()),
        ["Test", f"Average Optimal ({optimal_threshold:.4f})"] + list(valid_metrics['final_metrics'].values())
    ]
    
    if X_test is not None and y_test is not None:
        final_table.append(
            ["Test", f"Average Optimal ({optimal_threshold:.4f})"] + list(results['test']['final_metrics'].values())
        )
    
    # print("\nИтоговые метрики со средним оптимальным порогом:")
    # print(tabulate(final_table, headers="firstrow", floatfmt=".4f", tablefmt="grid"))

     # Формируем итоговый словарь в нужном формате
    model_package = {
        'model': model,
        'metrics': {
            'train': train_metrics['final_metrics'],
            'valid': valid_metrics['final_metrics'],
            'optimal_threshold': optimal_threshold
        },
        'features': list(X_train.columns),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if X_test is not None and y_test is not None:
        model_package['metrics']['test'] = results['test']['final_metrics']
    
    return model_package


# %% [markdown]
# # Интерпретация модели

# %% [markdown]
# ## SHAP

# %% [markdown]
# ### explain_model_shap

# %% [markdown]
# Функция **explain_model_shap**: 
# - Вычисляет SHAP-значения для интерпретации модели
# - Анализирует важность и направление влияния признаков
# - Визуализирует топ-N наиболее значимых признаков
#
# Входные данные:
# - X_train — датафрейм с признаками
# - model — обученная модель (RandomForest, XGBoost, LogisticRegression и др.)
# - sample_size — размер подвыборки для анализа (по умолчанию 2000)
# - top_n — количество топ-признаков для отображения
# - n_jobs — количество ядер для параллельных вычислений
#
# Выходные данные:
# - DataFrame с ранжированными признаками по важности
# - Визуализация важности признаков
#
# Особенности метода:
# - Автоматическое определение типа модели (TreeExplainer, LinearExplainer)
# - Поддержка многоклассовой классификации
# - Анализ направления влияния (Positive/Negative)
# - Сравнение с feature_importances_ модели
#
# Ключевые метрики:
# - Относительная важность признаков в %
# - Направление влияния на предсказание
# - Кумулятивная важность топ-признаков
#

# %%
def explain_model_shap(X_train, model, sample_size=2000, top_n=20, n_jobs = -1):
    """
    Оборачивает расчет SHAP-важности и визуализации признаков
    
    Параметры:
    ----------
    X_train : pd.DataFrame
        Датафрейм признаков
    model : sklearn/xgboost модель
        Обученная модель (RandomForest, LogisticRegression, XGB и др.)
    sample_size : int
        Размер случайной подвыборки
    top_n : int
        Кол-во признаков для отображения
    """
    try:
        total_start_time = time.time()
        model_type = type(model).__name__
        
        print(f"ℹ️ Model type: {model_type}")
        print(f"ℹ️ Number of classes: {getattr(model, 'n_classes_', 'unknown')}")
        
        # 1. Инициализация Explainer
        print("🔄 Initializing SHAP explainer...")
        explainer_start = time.time()
        if model_type in ['RandomForestClassifier', 'RandomForestRegressor', 
                          'XGBClassifier', 'XGBRegressor', 
                          'LGBMClassifier', 'LGBMRegressor']:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        elif model_type in ['LogisticRegression', 'LinearRegression']:
            explainer = shap.LinearExplainer(model, X_train)
        else:
            explainer = shap.Explainer(model, X_train)
        explainer_time = time.time() - explainer_start
        print(f"✅ SHAP explainer initialized in {timedelta(seconds=explainer_time)}")
        
        # 2. Подвыборка
        sample_size = min(sample_size, len(X_train))
        sample_idx = np.random.choice(X_train.index, size=sample_size, replace=False)
        X_sample = X_train.loc[sample_idx]

        print(f"\n🔄 Calculating SHAP values for {sample_size} samples...")
        shap_start = time.time()

        # Параллельная обработка
        n_jobs = n_jobs
        n_chunks = 4 * (os.cpu_count() or 1)

        def calc_chunk(chunk):
            return explainer.shap_values(chunk, approximate=True, check_additivity=False)

        chunks = np.array_split(X_sample, n_chunks)
        results = Parallel(n_jobs=n_jobs)(delayed(calc_chunk)(chunk) for chunk in chunks)

        # Объединение результатов
        if isinstance(results[0], list):
            shap_values = [np.concatenate([r[i] for r in results]) for i in range(len(results[0]))]
        else:
            shap_values = np.concatenate(results)

        shap_time = time.time() - shap_start
        print(f"✅ SHAP values calculated in {timedelta(seconds=shap_time)}")
        print(f"⏱ Average time per sample: {shap_time/sample_size:.4f} seconds")

        # 3. Обработка SHAP
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else np.mean(shap_values, axis=0)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        print(f"ℹ️ Processed SHAP values shape: {shap_values.shape}")

        # 4. Анализ важности
        print("\n🔄 Calculating feature importance...")
        analysis_start = time.time()
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'SHAP_Importance': np.abs(shap_values).mean(axis=0),
            'Direction': np.where(np.mean(shap_values, axis=0) > 0, 'Positive', 'Negative')
        })
        if hasattr(model, 'feature_importances_'):
            importance_df['Model_Importance'] = model.feature_importances_
            importance_df['Model_%'] = 100 * importance_df['Model_Importance'] / importance_df['Model_Importance'].max()

        importance_df['SHAP_%'] = 100 * importance_df['SHAP_Importance'] / importance_df['SHAP_Importance'].max()
        importance_df = importance_df.sort_values('SHAP_%', ascending=False)
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        importance_df['Cumulative_SHAP_%'] = importance_df['SHAP_%'].cumsum()
        analysis_time = time.time() - analysis_start
        print(f"✅ Feature analysis completed in {timedelta(seconds=analysis_time)}")

        # 5. Таблица
        print("\n🔍 Top Features by SHAP Importance:")
        display_cols = ['Rank', 'Feature', 'SHAP_%', 'Direction']
        if 'Model_%' in importance_df.columns:
            display_cols.append('Model_%')
        print(importance_df.head(top_n)[display_cols].to_markdown(index=False, floatfmt=".1f"))

        print("\n📊 Key Metrics:")
        print(f"• Top-5 features explain: {importance_df['Cumulative_SHAP_%'].iloc[4]:.1f}%")
        pos_count = (importance_df['Direction'] == 'Positive').sum()
        neg_count = (importance_df['Direction'] == 'Negative').sum()
        print(f"• Positive/Negative: {pos_count}/{neg_count}")

        # 6. Простая визуализация
        plt.figure(figsize=(10, min(6, top_n * 0.3)))
        colors = importance_df['Direction'].head(top_n).map({'Positive': 'tomato', 'Negative': 'dodgerblue'})
        plt.barh(importance_df['Feature'].head(top_n)[::-1], 
                 importance_df['SHAP_%'].head(top_n)[::-1],
                 color=colors[::-1])
        plt.title(f'Top {top_n} Features by SHAP')
        plt.xlabel('Relative SHAP Importance (%)')
        plt.tight_layout()
        plt.show()

        # 7. Общее время
        total_time = time.time() - total_start_time
        print(f"\n⏱ Total execution time: {timedelta(seconds=total_time)}")
        print("="*50)
        print("Time breakdown:")
        print(f"- Explainer init: {timedelta(seconds=explainer_time)}")
        print(f"- SHAP values: {timedelta(seconds=shap_time)} ({shap_time/total_time*100:.1f}%)")
        print(f"- Analysis: {timedelta(seconds=analysis_time)} ({analysis_time/total_time*100:.1f}%)")

        return importance_df

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if 'shap_values' in locals():
            print(f"SHAP values type: {type(shap_values)}")
            if hasattr(shap_values, 'shape'):
                print(f"SHAP values shape: {shap_values.shape}")
        print(f"X_train shape: {X_train.shape if X_train is not None else 'N/A'}")
        if hasattr(model, 'n_features_in_'):
            print(f"Model features: {model.n_features_in_}")
        return None


# %% [markdown]
# ## Permutation Importance

# %% [markdown]
# ### explain_model_permutation

# %% [markdown]
# **explain_model_permutation** Оценивает важность признаков с помощью Permutation Importance
#
# Вход:
# - X: DataFrame с признаками
# - y: целевая переменная
# - model: обученная модель (RandomForest, XGBoost и др.)
# - scoring: метрика оценки ('f1', 'accuracy', 'roc_auc')
# - n_repeats: количество повторов для стабильности
# - top_n: количество топ-признаков для отображения
# - random_state: seed для воспроизводимости
# - n_jobs: количество ядер для параллельных вычислений
#
# Выход:
# - DataFrame с важностью признаков (Feature, Mean Importance, Std, Significant, Rank)
# - Визуализация топ-признаков с доверительными интервалами
#

# %%
def explain_model_permutation(X, y, model, scoring='f1', n_repeats=5, top_n=20, random_state=3, n_jobs = 4):
    """
    Оценивает важность признаков с помощью Permutation Importance.
    
    Параметры:
    ----------
    X : pd.DataFrame
        Признаки (X_train или X_valid)
    y : pd.Series
        Целевая переменная
    model : обученная модель
        RandomForest, LogisticRegression, XGBoost и т.д.
    scoring : str
        Метрика (например, 'f1', 'accuracy', 'roc_auc')
    n_repeats : int
        Количество повторов для случайности
    top_n : int
        Кол-во признаков для отображения
    random_state : int
        Случайное зерно для воспроизводимости
    
    Возвращает:
    -----------
    pd.DataFrame — таблица важности признаков
    """
    try:
        print(f"ℹ️ Model type: {type(model).__name__}")
        print(f"ℹ️ Scoring metric: {scoring}")

        start_time = time.time()

        print("🔄 Calculating permutation importance...")
        result = permutation_importance(
            model, X, y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs
        )

        elapsed = time.time() - start_time
        print(f"✅ Completed in {timedelta(seconds=elapsed)}")

        # Формируем датафрейм
        importances_df = pd.DataFrame({
            'Feature': X.columns,
            'Mean Importance': result.importances_mean,
            'Std': result.importances_std
        })
        importances_df['Significant'] = importances_df['Mean Importance'] - 2 * importances_df['Std'] > 0
        importances_df = importances_df.sort_values(by='Mean Importance', ascending=False).reset_index(drop=True)
        importances_df['Rank'] = importances_df.index + 1

        print("\n🔍 Top Features by Permutation Importance:")
        display_cols = ['Rank', 'Feature', 'Mean Importance', 'Std', 'Significant']
        print(importances_df.head(top_n)[display_cols].to_markdown(index=False, floatfmt=".3f"))

        # Простая визуализация
        top_features = importances_df.head(top_n)
        plt.figure(figsize=(10, min(6, top_n * 0.3)))
        bars = plt.barh(top_features['Feature'][::-1], top_features['Mean Importance'][::-1],
                        xerr=top_features['Std'][::-1], color='mediumseagreen')
        plt.xlabel("Mean Importance")
        plt.title(f"Top {top_n} Features by Permutation Importance")
        plt.tight_layout()
        plt.show()

        return importances_df

    except Exception as e:
        print(f"❌ Error during permutation importance: {e}")
        return None


# %% [markdown]
# # Сигналы модели

# %% [markdown]
# **plot_predict_signals** Визуализирует OHLC график с истинными и предсказанными торговыми сигналами
#
# Аргументы:
# - df с колонками: 'Open','High','Low','Close','buy','sell'
# - y_pred: массив предсказаний модели (опционально)
# - pred_threshold: порог бинаризации предсказаний
# - start_idx: начальный индекс участка
# - end_idx: конечный индекс участка
#
# Отображает:
# - Линии OHLC цен
# - Истинные buy/sell сигналы (зеленые/красные маркеры)
# - Предсказанные buy сигналы модели (синие маркеры)

# %%
def plot_predict_signals(df, y_pred=None, pred_threshold=0.5, start_idx=200, end_idx=500):
    """
    Рисует график OHLC с подсветкой buy/sell сигналов и предсказаниями модели
    
    Args:
        df: DataFrame с колонками 'Open','High','Low','Close','buy','sell'
        y_pred: массив предсказаний модели (вероятности или бинарные)
        pred_threshold: порог для бинаризации предсказаний
        start_idx: начальный индекс участка
        end_idx: конечный индекс участка
    """
    if end_idx is None:
        end_idx = len(df)
    
    plot_data = df.iloc[start_idx:end_idx].copy()
    
    # Добавляем предсказания модели если они переданы
    if y_pred is not None:
        # Бинаризуем предсказания по порогу
        y_pred_binary = (y_pred[start_idx:end_idx] >= pred_threshold).astype(int)
        plot_data['model_buy'] = y_pred_binary
    
    plt.figure(figsize=(16, 8))
    
    # Рисуем все ценовые линии
    plt.plot(plot_data.index, plot_data['Close'], 'b-', label='Close', linewidth=1.5, alpha=0.8)
    plt.plot(plot_data.index, plot_data['Open'], 'g--', label='Open', linewidth=1, alpha=0.6)
    plt.plot(plot_data.index, plot_data['High'], 'c:', label='High', linewidth=1, alpha=0.6)
    plt.plot(plot_data.index, plot_data['Low'], 'm:', label='Low', linewidth=1, alpha=0.6)
    
    # Сигналы покупки (истинные)
    buy_signals = plot_data[plot_data['buy'] == 1]
    if not buy_signals.empty:
        plt.scatter(buy_signals.index, buy_signals['Close'], 
                   color='green', marker='^', s=120, label='True Buy', zorder=5, alpha=0.8)
    
    # Сигналы продажи (истинные)
    sell_signals = plot_data[plot_data['sell'] == 1]
    if not sell_signals.empty:
        plt.scatter(sell_signals.index, sell_signals['Close'], 
                   color='red', marker='v', s=120, label='True Sell', zorder=5, alpha=0.8)
    
    # Предсказания модели (если переданы)
    if y_pred is not None:
        model_buy_signals = plot_data[plot_data['model_buy'] == 1]
        if not model_buy_signals.empty:
            plt.scatter(model_buy_signals.index, model_buy_signals['Close'], 
                       color='blue', marker='^', s=100, label=f'Model Buy (≥{pred_threshold})', 
                       zorder=4, alpha=0.6, edgecolors='black', linewidth=1)
    
    plt.title(f'OHLC Prices with Buy/Sell Signals (Threshold: {pred_threshold})')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %% [markdown]
# # Бэктест модели

# %% [markdown]
# ## backtest_model

# %% [markdown]
# **backtest_model** Проводит пошаговый бэктест торговой модели с TP/SL и анализирует результаты
#
# Аргументы:
# - df: DataFrame с признаками и ценовыми данными
# - model: обученная модель машинного обучения
# - X_train: тренировочные данные, для необходимых признаков
# - threshold: порог для торговых сигналов
# - tp_pct: уровень тейк-профита (%)
# - rr: соотношение риск/прибыль
# - plot: флаг отображения графиков
#
# Возвращает:
# - Общую прибыль и счетчик TP/SL сделок
# - Максимальную серию убытков
# - Помесячную статистику
# - Таблицу всех сделок
# - Графики кривой капитала и месячной прибыли

# %%
def backtest_model(df, model, X_train, threshold=0.5, tp_pct=0.04, rr=2.0, plot=True):
    """
    Пошаговый бэктест торговой модели с TP и SL.
    После закрытия сделки следующая проверка начинается со следующей свечи.
    Результат: метрики, таблица сделок, помесячная статистика и кривая капитала.
    
    Args:
        df: DataFrame с данными для бэктеста
        model: обученная модель
        X_train: тренировочные данные (для получения feature_cols)
        threshold: порог для входа в сделку
        tp_pct: уровень тейк-профита в процентах
        rr: risk-reward ratio
        plot: строить ли графики
    """

    # Базовые колонки (исключаем те, что могут быть в df но не в фичах)
    base_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'buy', 'sell', 
                 'buy_strong', 'sell_strong', 'buy_noised', 'sell_noised']
    
    # Признаки берутся из X_train.columns
    feature_cols = [col for col in X_train.columns if col in df.columns]
    
    # Проверяем, что все фичи есть в df
    missing_features = set(X_train.columns) - set(df.columns)
    if missing_features:
        print(f"⚠️ Внимание: отсутствуют фичи в df: {missing_features}")
        print(f"Используем только доступные фичи: {feature_cols}")

    # Для SL
    sl_pct = tp_pct / rr

    # Предсказания (только если есть фичи)
    if feature_cols:
        preds = model.predict_proba(df[feature_cols])[:, 1]
        df = df.copy()
        df['pred'] = preds
    else:
        print("❌ Нет доступных фич для предсказания!")
        return None

    all_trades = []
    current_trade = None
    i = 0
    n = len(df)
    balance = [0]  # кривая капитала

    while i < n:
        row = df.iloc[i]

        # Если нет сделки — ищем вход
        if current_trade is None:
            if row['pred'] >= threshold:
                entry_price = row['Close']
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)

                current_trade = {
                    'entry_date': row['Date'],
                    'entry_price': entry_price,
                    'tp_price': tp_price,
                    'sl_price': sl_price
                }
        else:
            # Проверяем условия выхода
            if row['Low'] <= current_trade['sl_price']:
                current_trade['exit_date'] = row['Date']
                current_trade['outcome'] = 'SL'
                current_trade['profit_pct'] = -sl_pct
                all_trades.append(current_trade)
                balance.append(balance[-1] - sl_pct)
                current_trade = None
                i += 1  # проверка со следующей свечи
                continue

            if row['High'] >= current_trade['tp_price']:
                current_trade['exit_date'] = row['Date']
                current_trade['outcome'] = 'TP'
                current_trade['profit_pct'] = tp_pct
                all_trades.append(current_trade)
                balance.append(balance[-1] + tp_pct)
                current_trade = None
                i += 1
                continue

        i += 1

    # Закрываем последнюю сделку, если осталась
    if current_trade is not None:
        current_trade['exit_date'] = df['Date'].iloc[-1]
        current_trade['outcome'] = 'SL'
        current_trade['profit_pct'] = -sl_pct
        all_trades.append(current_trade)
        balance.append(balance[-1] - sl_pct)

    # В DataFrame для анализа
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

    # Метрики
    if not trades_df.empty:
        total_profit = trades_df['profit_pct'].sum() * 100
        tp_count = (trades_df['outcome'] == 'TP').sum()
        sl_count = (trades_df['outcome'] == 'SL').sum()
        max_sl_streak = (trades_df['outcome'] == 'SL').astype(int).groupby((trades_df['outcome'] != 'SL').cumsum()).sum().max()
        
        # Разбивка по месяцам
        trades_df['month'] = pd.to_datetime(trades_df['entry_date']).dt.to_period('M')
        monthly_profit = trades_df.groupby('month')['profit_pct'].sum() * 100
    else:
        total_profit = 0
        tp_count = 0
        sl_count = 0
        max_sl_streak = 0
        monthly_profit = pd.Series()

    results = {
        'total_profit': total_profit,
        'tp_count': tp_count,
        'sl_count': sl_count,
        'max_sl_streak': max_sl_streak,
        'monthly_profit': monthly_profit,
        'trades_df': trades_df,
        'feature_cols_used': feature_cols
    }

    # Построение графиков
    if plot and not trades_df.empty:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Кривая капитала с датами закрытия сделок
        trades_df['cum_profit'] = trades_df['profit_pct'].cumsum() * 100

        # Даты закрытия для оси X
        exit_dates = pd.to_datetime(trades_df['exit_date'])

        axes[0].plot(exit_dates, trades_df['cum_profit'], label='Equity Curve', color='blue')
        axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
        axes[0].set_title('Equity Curve')
        axes[0].set_xlabel('Exit Date')
        axes[0].set_ylabel('Cumulative Profit %')
        axes[0].legend()

        # Форматирование дат для удобочитаемости
        axes[0].xaxis.set_major_locator(mdates.AutoDateLocator())
        axes[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        fig.autofmt_xdate()

        # Прибыль по месяцам
        if not monthly_profit.empty:
            monthly_profit.plot(kind='bar', ax=axes[1], color='green')
            axes[1].set_title('Monthly Profit')
            axes[1].set_ylabel('Profit %')

        plt.tight_layout()
        plt.show()
    elif plot:
        print("⚠️ Нет сделок для построения графиков")

    return results


# %% [markdown]
# ## backtest_threshold_analysis

# %% [markdown]
# **backtest_threshold_analysis**
# Анализирует прибыльность модели по сетке порогов: для каждого **threshold** запускается пошаговый бэктест с **TP/SL** и собираются ключевые метрики.
#
# Аргументы:
# - df — DataFrame с OHLCV и признаками.
# - model — обученная модель.
# - X_train — тренировочный DataFrame (используется для выбора feature_cols).
# - thresholds — список порогов.
# - tp_pct — take-profit.
# - rr — risk/reward ratio (SL = TP / rr).
# - plot — рисовать график зависимости прибыли от порога.
#
# Возвращает
# - DataFrame с колонками: threshold, total_profit, tp_count, sl_count, total_trades, win_rate, avg_profit_per_trade.
# - При plot=True — график total_profit vs threshold с пометкой оптимального порога.

# %%
def backtest_threshold_analysis(df, model, X_train, thresholds=np.linspace(0.3, 0.9, 10), 
                               tp_pct=0.04, rr=2.0, plot=True):
    """
    Анализ прибыльности модели на различных пороговых значениях.
    
    """
    
    feature_cols = [col for col in X_train.columns if col in df.columns]
    
    if not feature_cols:
        print("❌ Нет доступных фич для предсказания!")
        return None
    
    preds = model.predict_proba(df[feature_cols])[:, 1]
    df = df.copy()
    df['pred'] = preds

    sl_pct = tp_pct / rr
    results = []
    
    for threshold in thresholds:
        print(f"Тестируем порог: {threshold:.3f}")
        
        all_trades = []
        current_trade = None
        i = 0
        n = len(df)

        while i < n:
            row = df.iloc[i]
            
            # Если есть открытая сделка - проверяем условия выхода
            if current_trade is not None:
                low_price = row['Low']
                high_price = row['High']
                
                # Проверяем SL
                if low_price <= current_trade['sl_price']:
                    current_trade['exit_date'] = row['Date']
                    current_trade['outcome'] = 'SL'
                    current_trade['profit_pct'] = -sl_pct
                    all_trades.append(current_trade)
                    current_trade = None
                    i += 1  # Пропускаем текущую свечу
                    continue
                
                # Проверяем TP
                if high_price >= current_trade['tp_price']:
                    current_trade['exit_date'] = row['Date']
                    current_trade['outcome'] = 'TP'
                    current_trade['profit_pct'] = tp_pct
                    all_trades.append(current_trade)
                    current_trade = None
                    i += 1  # Пропускаем текущую свечу
                    continue
            
            # Если нет сделки — ищем вход (только если current_trade is None)
            if current_trade is None:
                if row['pred'] >= threshold:
                    entry_price = row['Close']
                    tp_price = entry_price * (1 + tp_pct)
                    sl_price = entry_price * (1 - sl_pct)

                    current_trade = {
                        'entry_date': row['Date'],
                        'entry_price': entry_price,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'exit_date': None,
                        'outcome': None,
                        'profit_pct': 0
                    }
            
            i += 1

        # Закрываем последнюю сделку, если осталась
        if current_trade is not None:
            current_trade['exit_date'] = df['Date'].iloc[-1]
            current_trade['outcome'] = 'SL'
            exit_price = df['Close'].iloc[-1]
            pct_change = (exit_price - current_trade['entry_price']) / current_trade['entry_price']
            current_trade['profit_pct'] = pct_change
            all_trades.append(current_trade)

        # Остальная логика остается без изменений...
        trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

        if not trades_df.empty:
            total_profit = trades_df['profit_pct'].sum() * 100
            tp_count = (trades_df['outcome'] == 'TP').sum()
            sl_count = (trades_df['outcome'] == 'SL').sum()
            total_trades = len(trades_df)
            win_rate = tp_count / total_trades
            
            results.append({
                'threshold': threshold,
                'total_profit': total_profit,
                'tp_count': tp_count,
                'sl_count': sl_count,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_profit_per_trade': total_profit / total_trades
            })
        else:
            results.append({
                'threshold': threshold,
                'total_profit': 0,
                'tp_count': 0,
                'sl_count': 0,
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit_per_trade': 0
            })

    results_df = pd.DataFrame(results)
    
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['threshold'], results_df['total_profit'], 
                marker='o', linewidth=2, markersize=6, color='blue')
        plt.xlabel('Пороговое значение')
        plt.ylabel('Общая прибыль (%)')
        plt.title('Зависимость прибыли от порогового значения')
        plt.grid(True, alpha=0.3)
        
        if not results_df.empty:
            best_idx = results_df['total_profit'].idxmax()
            best_threshold = results_df.loc[best_idx, 'threshold']
            best_profit = results_df.loc[best_idx, 'total_profit']
            plt.axvline(x=best_threshold, color='red', linestyle='--', 
                       label=f'Оптимальный порог: {best_threshold:.3f}\nПрибыль: {best_profit:.2f}%')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return results_df

# %%
