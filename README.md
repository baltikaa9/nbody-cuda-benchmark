# CUDA N-Body: мног GPU benchmark

Benchmark моделирует N-body взаимодействия на CUDA и измеряет масштабирование вычислений при использовании нескольких GPU.

Текущая версия benchmark фиксирует одну вычислительную конфигурацию:

- тип данных: `float3`;
- kernel: `shared memory`;
- размер блока: `BS = 256`;
- количество итераций: `5`;
- передача данных: `host copy` и `GPU Direct`.

Сравнение `float3/float4` и `global/shared` memory в текущем benchmark не выполняется, чтобы сосредоточиться на масштабировании по количеству GPU.

## Сборка

Для сборки используется `nvcc` и OpenMP:

```bash
make build
```

Команда сборки:

```bash
nvcc -O3 -arch=native -Xcompiler -fopenmp nbody_bench.cu -o nbody_bench.o
```

## Запуск

Список доступных GPU:

```bash
make gpus
```

Запуск на выбранных GPU:

```bash
make run GPUS=0,1
```

По умолчанию используется:

```text
GPUS=0,1
```

Количество GPU можно переопределить:

```bash
make run GPUS=0,1,2,3
```

Программа последовательно выполняет тесты для:

```text
1 GPU
2 GPU
3 GPU
...
```

до количества доступных GPU.

## Режимы передачи данных

### Host copy

На каждой итерации каждый GPU получает полное состояние системы с host:

```text
Host → GPU
GPU считает свой диапазон тел
GPU → Host
```

После этого результаты собираются в памяти CPU и используются на следующей итерации.

### GPU Direct

После начальной загрузки данных обмен выполняется напрямую между GPU через CUDA P2P:

```text
GPU 0 ───────► GPU 1
GPU 1 ───────► GPU 0
```

Используются:

```cpp
cudaDeviceCanAccessPeer()
cudaDeviceEnablePeerAccess()
cudaMemcpyPeer()
```

Если выбранные GPU не поддерживают P2P, режим `GPU Direct` пропускается.

## Результаты

Benchmark сохраняет результаты в:

```text
benchmark_results.csv
```

Формат CSV:

```text
variant,gpus,type,N,BS,avg_ms,ms_per_pair,tflops
```

Пример вариантов:

```text
multi_host,float3,...
multi_direct,float3,...
```

Основные метрики:

- `avg_ms` — среднее время одной итерации;
- `ms_per_pair` — время на одну пару взаимодействующих тел;
- `tflops` — оценка производительности в TFLOP/s.

## Построение графиков

Графики строятся через `uv`:

```bash
make plot GPUS=0,1
```

Будут созданы:

```text
plot_latency.png
plot_throughput.png
plot_scaling.png
plot_speedup.png
plot_efficiency.png
plot_host_vs_direct.png
```

### `plot_latency.png`

Зависимость `ms_per_pair` от размера задачи `N`.

Меньшее значение означает лучшую производительность.

### `plot_throughput.png`

Производительность в `TFLOP/s` для разных размеров `N`.

Большее значение означает лучшую производительность.

### `plot_scaling.png`

Изменение задержки и производительности при увеличении количества GPU для максимального значения `N`.

### `plot_speedup.png`

Ускорение относительно одного GPU:

```text
S(p) = T(1) / T(p)
```

На графике также отображается идеальная линия масштабирования:

```text
S(p) = p
```

### `plot_efficiency.png`

Эффективность использования GPU:

```text
E(p) = S(p) / p × 100%
```

Идеальная эффективность равна `100%`.

### `plot_host_vs_direct.png`

Сравнение способов передачи данных:

```text
host_direct_speedup = T(host copy) / T(GPU Direct)
```

Значение больше `1` означает, что `GPU Direct` быстрее `host copy`.

## Очистка результатов

Удалить CSV и PNG-файлы:

```bash
make clean-data
```

Удалить результаты и бинарный файл:

```bash
make clean
```

> Если PNG-файлы используются как изображения в этом README и добавлены в Git, не запускайте `make clean-data` без необходимости: команда удаляет все `*.png` и `*.csv` в каталоге проекта.
