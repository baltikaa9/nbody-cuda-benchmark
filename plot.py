import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Чистый стиль без лишних украшательств
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10, 
    'axes.titlesize': 12,
    'figure.facecolor': 'white',  # Чисто белый фон окна
    'axes.facecolor': 'white'    # Чисто белый фон графиков
})

try:
    df = pd.read_csv("benchmark_results.csv")
except FileNotFoundError:
    print("Ошибка: Файл benchmark_results.csv не найден.")
    exit(1)

v_types = ['float3', 'float4', 'double3', 'double4']
n_vals = sorted(df['N'].unique())

# Используем layout='constrained' для автоматических отступов
fig = plt.figure(figsize=(20, 11), layout='constrained')
gs = fig.add_gridspec(3, 4)

def plot_type_metric(ax, vtype, metric, title, ylabel):
    df_type = df[df['type'] == vtype]
    sns.lineplot(data=df_type, x='N', y=metric, hue='BS', 
                 marker='o', palette='tab10', ax=ax)
    ax.set_title(f"{vtype}: {title}", fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(n) for n in n_vals])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("N")
    ax.legend(title='BS', fontsize='8', loc='best')

# СТРОКА 1: Задержка
for i, vt in enumerate(v_types):
    ax = fig.add_subplot(gs[0, i])
    plot_type_metric(ax, vt, 'ms_per_pair', 'Задержка', 'мс/пара')

# СТРОКА 2: Производительность
for i, vt in enumerate(v_types):
    ax = fig.add_subplot(gs[1, i])
    plot_type_metric(ax, vt, 'gints', 'Производительность', 'GInt/s')

# СТРОКА 3: СРАВНЕНИЯ
ax_comp_ms = fig.add_subplot(gs[2, 0:2])
ax_comp_gi = fig.add_subplot(gs[2, 2:4])

for vt in v_types:
    df_vt = df[df['type'] == vt]
    best_ms = df_vt.loc[df_vt.groupby('N')['ms_per_pair'].idxmin()].sort_values('N')
    best_gi = df_vt.loc[df_vt.groupby('N')['gints'].idxmax()].sort_values('N')
    
    ax_comp_ms.plot(best_ms['N'], best_ms['ms_per_pair'], '-s', label=vt, linewidth=2)
    ax_comp_gi.plot(best_gi['N'], best_gi['gints'], '-o', label=vt, linewidth=2)

ax_comp_ms.set_title('СРАВНЕНИЕ ЗАДЕРЖКИ (Best BS)', fontweight='bold')
ax_comp_gi.set_title('СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ (Best BS)', fontweight='bold')

for ax in [ax_comp_ms, ax_comp_gi]:
    ax.set_xscale('log', base=2)
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(n) for n in n_vals])
    ax.legend(title='Тип')
    ax.set_xlabel("N")

fig.suptitle('Отчет производительности CUDA N-Body', fontsize=18, fontweight='bold')

plt.savefig("plot.png", dpi=150)
print("Чистый график сохранен в plot.png")
plt.show()