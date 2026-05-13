import pandas as pd
import matplotlib.pyplot as plt

# Загружаем данные
try:
    df = pd.read_csv("benchmark_results.csv")
except FileNotFoundError:
    print("Файл benchmark_results.csv не найден! Сначала запустите CUDA-программу.")
    exit(1)

# Определяем типы векторов и метрики
types = ['float3', 'float4', 'double3', 'double4']
metrics = [
    ('ms_per_pair', 'Время на пару (ms) [Меньше = Лучше]', False), 
    ('gints', 'Производительность (GInt/s) [Больше = Лучше]', True)
]

# Создаем сетку графиков 2х5 (2 строки = 2 метрики, 5 колонок = 4 типа + 1 сравнение)
fig, axes = plt.subplots(2, 5, figsize=(28, 12))
fig.suptitle('Детальный анализ производительности N-Body', fontsize=18, fontweight='bold')

for m_idx, (metric, ylabel, is_higher_better) in enumerate(metrics):
    
    # Графики 1-4: Зависимость метрики от N для каждого BS по типам векторов
    for t_idx, vtype in enumerate(types):
        ax = axes[m_idx, t_idx]
        df_type = df[df['type'] == vtype]
        
        # Строим линию для каждого размера блока (BS)
        for bs in sorted(df['BS'].unique()):
            df_bs = df_type[df_type['BS'] == bs].sort_values(by='N')
            if not df_bs.empty:
                ax.plot(df_bs['N'], df_bs[metric], marker='o', markersize=5, label=f'BS={bs}')
        
        ax.set_title(f"{vtype}\n{metric}")
        ax.set_xlabel("Число тел (N)")
        ax.set_ylabel(ylabel)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(df['N'].unique()))
        ax.set_xticklabels([str(n) for n in sorted(df['N'].unique())], rotation=45)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8)

    # График 5: Итоговое сравнение типов (берем ЛУЧШИЙ результат среди всех BS для каждого N)
    ax_comp = axes[m_idx, 4]
    for vtype in types:
        df_type = df[df['type'] == vtype]
        
        # Выбираем лучший BS для каждого N
        if is_higher_better:
            best_df = df_type.loc[df_type.groupby('N')[metric].idxmax()].sort_values(by='N')
        else:
            best_df = df_type.loc[df_type.groupby('N')[metric].idxmin()].sort_values(by='N')
        
        ax_comp.plot(best_df['N'], best_df[metric], marker='s', linewidth=2, markersize=7, label=vtype)
        
    ax_comp.set_title(f"СРАВНЕНИЕ ТИПОВ\n(Лучшие BS) - {metric}")
    ax_comp.set_xlabel("Число тел (N)")
    ax_comp.set_ylabel(ylabel)
    ax_comp.set_xscale('log', base=2)
    ax_comp.set_xticks(sorted(df['N'].unique()))
    ax_comp.set_xticklabels([str(n) for n in sorted(df['N'].unique())], rotation=45)
    ax_comp.grid(True, linestyle='-', alpha=0.8)
    ax_comp.legend(fontsize=10, title="Vector Type")

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Оставляем место для главного заголовка
plt.savefig("benchmark_10_plots.png", dpi=300, bbox_inches='tight')
print("Все 10 графиков успешно сохранены в файл benchmark_10_plots.png")
plt.show()