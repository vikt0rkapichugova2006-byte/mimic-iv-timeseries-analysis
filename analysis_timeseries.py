import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os

os.makedirs('images_ts', exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ (имитация MIMIC-IV)
np.random.seed(42)
n_hours = 1200
dates = pd.date_range(start='2023-01-01', periods=n_hours, freq='h')

t = np.arange(n_hours)
seasonality = np.sin(2 * np.pi * t / 24)

data = {
    'charttime': dates,
    'HeartRate': 80 + 5 * seasonality + np.random.normal(0, 3, n_hours),
    'SysBP': 120 + 8 * seasonality + np.random.normal(0, 5, n_hours),
    'DiaBP': 70 + 4 * seasonality + np.random.normal(0, 3, n_hours),
    'RespRate': 18 + 1 * seasonality + np.random.normal(0, 1.5, n_hours),
    'Temperature': 36.8 + 0.3 * seasonality + np.random.normal(0, 0.2, n_hours),
    'SpO2': 98 - 0.5 * seasonality + np.random.normal(0, 0.5, n_hours)
}

df_ts = pd.DataFrame(data)
df_ts['charttime'] = pd.to_datetime(df_ts['charttime'])
df_ts.set_index('charttime', inplace=True)

df_ts.loc[df_ts.sample(frac=0.017).index, 'HeartRate'] = np.nan
df_ts.loc[df_ts.sample(frac=0.021).index, 'SysBP'] = np.nan
df_ts.loc[df_ts.sample(frac=0.021).index, 'DiaBP'] = np.nan
df_ts.loc[df_ts.sample(frac=0.025).index, 'RespRate'] = np.nan
df_ts.loc[df_ts.sample(frac=0.083).index, 'Temperature'] = np.nan
df_ts.loc[df_ts.sample(frac=0.008).index, 'SpO2'] = np.nan

channels = ['HeartRate', 'SysBP', 'DiaBP', 'RespRate', 'Temperature', 'SpO2']

print("Данные сгенерированы. Приступаю к построению графиков...")

# РИСУНОК 2.1. Первые строки данных (таблица)
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
table_data = df_ts.head(10).reset_index().values
ax.table(
    cellText=table_data,
    colLabels=['charttime'] + channels,
    loc='center',
    cellLoc='center'
)
ax.set_title('Рисунок 2.1. Первые строки многомерного временного ряда', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('images_ts/fig2_1_head_table.png', dpi=300, bbox_inches='tight')
plt.show()
print("Рисунок 2.1 сохранён.")

# РИСУНОК 2.2. Графики шести каналов временного ряда
fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
colors = ['steelblue', 'coral', 'seagreen', 'goldenrod', 'purple', 'teal']

for i, col in enumerate(channels):
    axes[i].plot(df_ts.index, df_ts[col], color=colors[i], linewidth=0.8, alpha=0.8)
    axes[i].set_ylabel(col, fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.4)

plt.xlabel('Время (charttime)', fontsize=11)
plt.suptitle('Рисунок 2.2. Графики шести каналов временного ряда', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('images_ts/fig2_2_channels.png', dpi=300, bbox_inches='tight')
plt.show()
print("Рисунок 2.2 сохранён.")

# РИСУНОК 2.3. Диаграммы размаха (Boxplots) для всех каналов
plt.figure(figsize=(12, 6))
df_melted = df_ts.melt(var_name='Канал', value_name='Значение')
sns.boxplot(data=df_melted, x='Канал', y='Значение', palette='pastel')
plt.title('Рисунок 2.3. Диаграммы размаха для всех каналов', fontsize=13)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('images_ts/fig2_3_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()
print("Рисунок 2.3 сохранён.")

# РИСУНОК 2.4. Сравнение диапазонов значений (среднее ± std)
stats = df_ts.describe().loc[['mean', 'std']].T
plt.figure(figsize=(10, 6))
plt.bar(stats.index, stats['mean'], yerr=stats['std'],
        capsize=6, color='skyblue', edgecolor='navy', alpha=0.8)
plt.title('Рисунок 2.4. Сравнение средних значений и разброса каналов', fontsize=13)
plt.ylabel('Значение')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('images_ts/fig2_4_ranges.png', dpi=300, bbox_inches='tight')
plt.show()
print("Рисунок 2.4 сохранён.")

# РИСУНОК 2.5. Тепловая карта корреляций
plt.figure(figsize=(8, 7))
corr_matrix = df_ts[channels].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Рисунок 2.5. Тепловая карта корреляций между каналами', fontsize=13)
plt.tight_layout()
plt.savefig('images_ts/fig2_5_corr.png', dpi=300, bbox_inches='tight')
plt.show()
print("Рисунок 2.5 сохранён.")

# РИСУНОК 2.6. Декомпозиция канала HeartRate
hr_clean = df_ts['HeartRate'].dropna().asfreq('h').interpolate(method='linear')
decomposition = seasonal_decompose(hr_clean, model='additive', period=24)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
axes[0].plot(decomposition.observed, color='steelblue', linewidth=0.8)
axes[0].set_ylabel('Наблюдаемое')
axes[1].plot(decomposition.trend, color='orange', linewidth=0.8)
axes[1].set_ylabel('Тренд')
axes[2].plot(decomposition.seasonal, color='seagreen', linewidth=0.8)
axes[2].set_ylabel('Сезонность')
axes[3].plot(decomposition.resid, color='coral', linewidth=0.8)
axes[3].set_ylabel('Остатки')

plt.xlabel('Время')
plt.suptitle('Рисунок 2.6. Декомпозиция канала HeartRate', fontsize=14)
plt.tight_layout()
plt.savefig('images_ts/fig2_6_decomposition.png', dpi=300, bbox_inches='tight')
plt.show()
print(" Рисунок 2.6 сохранён.")

# РИСУНОК 2.7. Гистограмма распределения остатков
plt.figure(figsize=(10, 6))
sns.histplot(decomposition.resid.dropna(), bins=30, kde=True, color='salmon', alpha=0.7)
plt.title('Рисунок 2.7. Гистограмма распределения остатков канала HeartRate', fontsize=13)
plt.xlabel('Значение остатка')
plt.ylabel('Частота')
plt.tight_layout()
plt.savefig('images_ts/fig2_7_resid_hist.png', dpi=300, bbox_inches='tight')
plt.show()
print("Рисунок 2.7 сохранён.")

# ВЫВОД СТАТИСТИКИ (для таблицы 2.3 и 2.4 в документе)
print("\n" + "="*60)
print(" ТАБЛИЦА 2.3 — Описательные статистики:")
print(df_ts[channels].describe().round(1))

print("\n📊 ТАБЛИЦА 2.4 — Доля пропусков по каналам:")
missing_pct = (df_ts[channels].isnull().sum() / len(df_ts) * 100).round(1)
print(missing_pct)

print("\n Все 7 рисунков сохранены в папку 'images_ts/'!")
