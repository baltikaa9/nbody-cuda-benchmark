import pandas as pd
import matplotlib.pyplot as plt

# Загружаем данные
try:
    df = pd.read_csv("benchmark_results.csv")
except FileNotFoundError:
    print("Файл benchmark_results.csv не найден! Сначала запустите CUDA-программу.")
    exit(1)

# Для каждого размера N и каждого типа вектора найдем лучший результат
# (максимальный GInt/s), так как размер блока (BS) может влиять по-разному
best_perf = df.loc[df.groupby(['type', 'N'])['gints'].idxmax()]

# Настройки графика
plt.figure(figsize=(10, 6))
colors = {'float3': 'blue', 'float4': 'cyan', 'double3': 'red', 'double4': 'orange'}
markers = {'float3': 'o', 'float4': 's', 'double3': '^', 'double4': 'D'}

# Рисуем линии для каждого типа
for vtype in df['type'].unique():
    subset = best_perf[best_perf['type'] == vtype].sort_values(by='N')
    plt.plot(subset['N'], subset['gints'], 
             label=vtype, 
             color=colors.get(vtype, 'black'),
             marker=markers.get(vtype, 'o'),
             linewidth=2, markersize=8)

# Оформление графика
plt.title('N-Body Simulation Performance (Best Block Size)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Bodies (N)', fontsize=12)
plt.ylabel('Performance (GInt/s)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Vector Type', fontsize=10)

# Логарифмическая шкала по X полезна, так как N растет в степени двойки
plt.xscale('log', base=2)
plt.xticks(df['N'].unique(), labels=[str(n) for n in sorted(df['N'].unique())])

plt.tight_layout()

# Сохраняем и показываем
plt.savefig("benchmark_plot.png", dpi=300)
print("График сохранен как benchmark_plot.png")
plt.show()