#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ════════════════════════════════════════════════════════════════════
// Traits: scalar type, EPS, DT, display name — по одному на тип
// ════════════════════════════════════════════════════════════════════
template<typename VEC> struct Traits;

#define MAKE_TRAITS(VEC, SCALAR, EPS_VAL, DT_VAL, NAME)  \
template<> struct Traits<VEC> {                            \
    using S = SCALAR;                                       \
    static constexpr S EPS = EPS_VAL;                       \
    static constexpr S DT  = DT_VAL;                        \
    static const char* name() { return NAME; }              \
};

MAKE_TRAITS(float3,  float,  1e-4f, 0.01f, "float3" )
MAKE_TRAITS(float4,  float,  1e-4f, 0.01f, "float4" )
MAKE_TRAITS(double3, double, 1e-4,  0.01,  "double3")
MAKE_TRAITS(double4, double, 1e-4,  0.01,  "double4")

// ════════════════════════════════════════════════════════════════════
// Device helpers
// ════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float  my_rsqrt(float  x) { return rsqrtf(x); }
__device__ __forceinline__ double my_rsqrt(double x) { return rsqrt(x);  }

template<typename VEC> __device__ __forceinline__ VEC vec_zero();
template<> __device__ __forceinline__ float3  vec_zero<float3>()  { return make_float3 (0.f, 0.f, 0.f);      }
template<> __device__ __forceinline__ float4  vec_zero<float4>()  { return make_float4 (0.f, 0.f, 0.f, 0.f); }
template<> __device__ __forceinline__ double3 vec_zero<double3>() { return make_double3(0.0, 0.0, 0.0);       }
template<> __device__ __forceinline__ double4 vec_zero<double4>() { return make_double4(0.0, 0.0, 0.0, 0.0); }

// ════════════════════════════════════════════════════════════════════
// N-body kernel — наивный (глобальная память)
// ════════════════════════════════════════════════════════════════════
template<typename VEC>
__global__ void integrateBodies(
    VEC* __restrict__       newPos,
    VEC* __restrict__       newVel,
    const VEC* __restrict__ oldPos,
    const VEC* __restrict__ oldVel,
    int N,
    typename Traits<VEC>::S dt
) {
    using S = typename Traits<VEC>::S;
    const S EPS2 = Traits<VEC>::EPS * Traits<VEC>::EPS;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    VEC pos = oldPos[idx];
    VEC f   = vec_zero<VEC>();

    for (int i = 0; i < N; i++) {
        VEC pi = oldPos[i];
        S dx = pi.x - pos.x;
        S dy = pi.y - pos.y;
        S dz = pi.z - pos.z;

        S dist2 = dx*dx + dy*dy + dz*dz + EPS2;
        S inv   = my_rsqrt(dist2);
        S s     = inv * inv * inv;

        f.x += dx * s;
        f.y += dy * s;
        f.z += dz * s;
    }

    VEC vel = oldVel[idx];
    vel.x += f.x * dt;  vel.y += f.y * dt;  vel.z += f.z * dt;
    pos.x += vel.x * dt; pos.y += vel.y * dt; pos.z += vel.z * dt;

    newPos[idx] = pos;
    newVel[idx] = vel;
}

// ════════════════════════════════════════════════════════════════════
// N-body kernel — с разделяемой памятью (тайлинг)
// ════════════════════════════════════════════════════════════════════
template<typename VEC, int BLOCK_SIZE>
__global__ void integrateBodiesShared(
    VEC* __restrict__       newPos,
    VEC* __restrict__       newVel,
    const VEC* __restrict__ oldPos,
    const VEC* __restrict__ oldVel,
    int N,
    typename Traits<VEC>::S dt
) {
    using S = typename Traits<VEC>::S;
    const S EPS2 = Traits<VEC>::EPS * Traits<VEC>::EPS;

    __shared__ VEC sharedPos[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    VEC pos, vel;
    VEC f = vec_zero<VEC>();

    if (idx < N) {
        pos = oldPos[idx];
    }

    // Проходим по всем тайлам
    for (int tile = 0; tile < numTiles; tile++) {
        int tileStart = tile * BLOCK_SIZE;
        int loadIdx = tileStart + threadIdx.x;

        // Кооперативная загрузка тайла в shared memory
        // Заполняем нулями, если за пределами N — иначе мусор
        sharedPos[threadIdx.x] = (loadIdx < N) ? oldPos[loadIdx] : vec_zero<VEC>();
        __syncthreads();

        int tileEnd = min(tileStart + BLOCK_SIZE, N);
        int tileSize = tileEnd - tileStart;

        // Вычисляем взаимодействия с телами из текущего тайла
        if (idx < N) {
            for (int i = 0; i < tileSize; i++) {
                VEC pi = sharedPos[i];
                S dx = pi.x - pos.x;
                S dy = pi.y - pos.y;
                S dz = pi.z - pos.z;

                S dist2 = dx*dx + dy*dy + dz*dz + EPS2;
                S inv   = my_rsqrt(dist2);
                S s     = inv * inv * inv;

                f.x += dx * s;
                f.y += dy * s;
                f.z += dz * s;
            }
        }

        __syncthreads();
    }

    if (idx < N) {
        vel = oldVel[idx];
        vel.x += f.x * dt;  vel.y += f.y * dt;  vel.z += f.z * dt;
        pos.x += vel.x * dt; pos.y += vel.y * dt; pos.z += vel.z * dt;

        newPos[idx] = pos;
        newVel[idx] = vel;
    }
}

// ════════════════════════════════════════════════════════════════════
// Макросы для проверки ошибок
// ════════════════════════════════════════════════════════════════════
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

static bool kernelOk(const char* label) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("%-8s  SKIP (%s)\n", label, cudaGetErrorString(e));
        return false;
    }
    return true;
}

template<typename VEC>
void randomInit(VEC* a, int N) {
    using S = typename Traits<VEC>::S;
    for (int i = 0; i < N; i++) {
        a[i].x = (S)(rand() / (double)RAND_MAX - 0.5);
        a[i].y = (S)(rand() / (double)RAND_MAX - 0.5);
        a[i].z = (S)(rand() / (double)RAND_MAX - 0.5);
    }
}

// ════════════════════════════════════════════════════════════════════
// Вспомогательная функция для диспетчеризации по размеру блока
// ════════════════════════════════════════════════════════════════════
template<typename VEC>
void launchSharedKernel(int grid, int BS,
                        VEC* d_pos_out, VEC* d_vel_out,
                        const VEC* d_pos_in, const VEC* d_vel_in,
                        int N, typename Traits<VEC>::S dt)
{
    switch (BS) {
        case 32:   integrateBodiesShared<VEC, 32><<<grid, BS>>>(d_pos_out, d_vel_out, d_pos_in, d_vel_in, N, dt); break;
        case 64:   integrateBodiesShared<VEC, 64><<<grid, BS>>>(d_pos_out, d_vel_out, d_pos_in, d_vel_in, N, dt); break;
        case 128:  integrateBodiesShared<VEC, 128><<<grid, BS>>>(d_pos_out, d_vel_out, d_pos_in, d_vel_in, N, dt); break;
        case 256:  integrateBodiesShared<VEC, 256><<<grid, BS>>>(d_pos_out, d_vel_out, d_pos_in, d_vel_in, N, dt); break;
        case 512:  integrateBodiesShared<VEC, 512><<<grid, BS>>>(d_pos_out, d_vel_out, d_pos_in, d_vel_in, N, dt); break;
        case 1024: integrateBodiesShared<VEC, 1024><<<grid, BS>>>(d_pos_out, d_vel_out, d_pos_in, d_vel_in, N, dt); break;
        default:
            fprintf(stderr, "Unsupported block size: %d\n", BS);
            exit(EXIT_FAILURE);
    }
}

// ════════════════════════════════════════════════════════════════════
// Запуск одного эксперимента (наивный)
// ════════════════════════════════════════════════════════════════════
template<typename VEC>
void runBenchmark(int N, int BS, int iters,
                  cudaEvent_t evStart, cudaEvent_t evStop, FILE* csv_file)
{
    using S = typename Traits<VEC>::S;

    VEC* h_pos = (VEC*)calloc(N, sizeof(VEC));
    VEC* h_vel = (VEC*)calloc(N, sizeof(VEC));
    if (!h_pos || !h_vel) { fputs("OOM\n", stderr); exit(1); }

    randomInit<VEC>(h_pos, N);
    randomInit<VEC>(h_vel, N);

    VEC *d_pos[2], *d_vel[2];
    CUDA_CHECK(cudaMalloc(&d_pos[0], N * sizeof(VEC)));
    CUDA_CHECK(cudaMalloc(&d_vel[0], N * sizeof(VEC)));
    CUDA_CHECK(cudaMalloc(&d_pos[1], N * sizeof(VEC)));
    CUDA_CHECK(cudaMalloc(&d_vel[1], N * sizeof(VEC)));

    CUDA_CHECK(cudaMemcpy(d_pos[0], h_pos, N*sizeof(VEC), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel[0], h_vel, N*sizeof(VEC), cudaMemcpyHostToDevice));

    int grid = (N + BS - 1) / BS;

    integrateBodies<VEC><<<grid, BS>>>(
        d_pos[1], d_vel[1], d_pos[0], d_vel[0], N, Traits<VEC>::DT);

    char label[64];
    snprintf(label, sizeof(label), "%s N=%d BS=%d", Traits<VEC>::name(), N, BS);

    bool ok = kernelOk(label);
    if (ok) CUDA_CHECK(cudaDeviceSynchronize());

    if (ok) {
        CUDA_CHECK(cudaEventRecord(evStart));
        int cur = 0;
        for (int it = 0; it < iters; it++, cur ^= 1)
            integrateBodies<VEC><<<grid, BS>>>(
                d_pos[cur^1], d_vel[cur^1],
                d_pos[cur],   d_vel[cur],
                N, Traits<VEC>::DT);
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));

        if (kernelOk(label)) {
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
            float avg_ms  = ms / iters;
            long long pairs = (long long)N * N;
            double flops = (double)pairs * 18 / (avg_ms * 1e-3) / 1e12;

            printf("%-8s %7d %5d %12.3f %14.3e %10.3f\n",
                   Traits<VEC>::name(), N, BS, avg_ms,
                   avg_ms / (double)pairs, flops);

            if (csv_file) {
                fprintf(csv_file, "global,%s,%d,%d,%.3f,%.3e,%.3f\n",
                        Traits<VEC>::name(), N, BS, avg_ms, avg_ms / (double)pairs, flops);
                fflush(csv_file);
            }
        }
    }

    cudaFree(d_pos[0]); cudaFree(d_pos[1]);
    cudaFree(d_vel[0]); cudaFree(d_vel[1]);
    free(h_pos); free(h_vel);
}

// ════════════════════════════════════════════════════════════════════
// Запуск одного эксперимента (разделяемая память)
// ════════════════════════════════════════════════════════════════════
template<typename VEC>
void runBenchmarkShared(int N, int BS, int iters,
                        cudaEvent_t evStart, cudaEvent_t evStop, FILE* csv_file)
{
    using S = typename Traits<VEC>::S;

    VEC* h_pos = (VEC*)calloc(N, sizeof(VEC));
    VEC* h_vel = (VEC*)calloc(N, sizeof(VEC));
    if (!h_pos || !h_vel) { fputs("OOM\n", stderr); exit(1); }

    randomInit<VEC>(h_pos, N);
    randomInit<VEC>(h_vel, N);

    VEC *d_pos[2], *d_vel[2];
    CUDA_CHECK(cudaMalloc(&d_pos[0], N * sizeof(VEC)));
    CUDA_CHECK(cudaMalloc(&d_vel[0], N * sizeof(VEC)));
    CUDA_CHECK(cudaMalloc(&d_pos[1], N * sizeof(VEC)));
    CUDA_CHECK(cudaMalloc(&d_vel[1], N * sizeof(VEC)));

    CUDA_CHECK(cudaMemcpy(d_pos[0], h_pos, N*sizeof(VEC), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel[0], h_vel, N*sizeof(VEC), cudaMemcpyHostToDevice));

    int grid = (N + BS - 1) / BS;

    // Прогрев
    launchSharedKernel<VEC>(grid, BS,
        d_pos[1], d_vel[1], d_pos[0], d_vel[0], N, Traits<VEC>::DT);

    char label[64];
    snprintf(label, sizeof(label), "shm_%s N=%d BS=%d", Traits<VEC>::name(), N, BS);

    bool ok = kernelOk(label);
    if (ok) CUDA_CHECK(cudaDeviceSynchronize());

    if (ok) {
        CUDA_CHECK(cudaEventRecord(evStart));
        int cur = 0;
        for (int it = 0; it < iters; it++, cur ^= 1)
            launchSharedKernel<VEC>(grid, BS,
                d_pos[cur^1], d_vel[cur^1],
                d_pos[cur],   d_vel[cur],
                N, Traits<VEC>::DT);
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));

        if (kernelOk(label)) {
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
            float avg_ms  = ms / iters;
            long long pairs = (long long)N * N;
            double flops = (double)pairs * 18 / (avg_ms * 1e-3) / 1e12;

            printf("%-8s %7d %5d %12.3f %14.3e %10.3f\n",
                   Traits<VEC>::name(), N, BS, avg_ms,
                   avg_ms / (double)pairs, flops);

            if (csv_file) {
                fprintf(csv_file, "shared,%s,%d,%d,%.3f,%.3e,%.3f\n",
                        Traits<VEC>::name(), N, BS, avg_ms, avg_ms / (double)pairs, flops);
                fflush(csv_file);
            }
        }
    }

    cudaFree(d_pos[0]); cudaFree(d_pos[1]);
    cudaFree(d_vel[0]); cudaFree(d_vel[1]);
    free(h_pos); free(h_vel);
}

// ════════════════════════════════════════════════════════════════════
// main
// ════════════════════════════════════════════════════════════════════
int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (SM %d.%d, %zu MB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem >> 20);

    // Открываем CSV-файл (теперь с колонкой variant)
    FILE* csv_file = fopen("benchmark_results.csv", "w");
    if (csv_file) {
        fprintf(csv_file, "variant,type,N,BS,avg_ms,ms_per_pair,tflops\n");
    } else {
        fprintf(stderr, "Warning: Could not open benchmark_results.csv for writing\n");
    }

    // ════════════════════════════════════════════════════════════════
    // Таблица 1: Глобальная память
    // ════════════════════════════════════════════════════════════════
    printf("═══ Global memory ═══\n");
    printf("%-8s %7s %5s %12s %14s %10s\n",
           "TYPE", "N", "BS", "avg_ms", "ms/pair", "TFLOP/s");
    printf("─────────────────────────────────────────────────────────────────\n");

    const int N_list[]  = {4096, 8192, 16384, 32768, 65536};
    const int BS_list[] = {32, 64, 128, 256, 512, 1024};
    const int iters     = 5;

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    for (int ni = 0; ni < 5; ni++) {
        int N = N_list[ni];
        for (int bi = 0; bi < 6; bi++) {
            int BS = BS_list[bi];
            runBenchmark<float3> (N, BS, iters, evStart, evStop, csv_file);
            runBenchmark<float4> (N, BS, iters, evStart, evStop, csv_file);
            runBenchmark<double3>(N, BS, iters, evStart, evStop, csv_file);
            runBenchmark<double4>(N, BS, iters, evStart, evStop, csv_file);
        }
        puts("");
    }

    // ════════════════════════════════════════════════════════════════
    // Таблица 2: Разделяемая память
    // ════════════════════════════════════════════════════════════════
    printf("\n═══ Shared memory ═══\n");
    printf("%-8s %7s %5s %12s %14s %10s\n",
           "TYPE", "N", "BS", "avg_ms", "ms/pair", "TFLOP/s");
    printf("─────────────────────────────────────────────────────────────────\n");

    for (int ni = 0; ni < 5; ni++) {
        int N = N_list[ni];
        for (int bi = 0; bi < 6; bi++) {
            int BS = BS_list[bi];
            runBenchmarkShared<float3> (N, BS, iters, evStart, evStop, csv_file);
            runBenchmarkShared<float4> (N, BS, iters, evStart, evStop, csv_file);
            runBenchmarkShared<double3>(N, BS, iters, evStart, evStop, csv_file);
            runBenchmarkShared<double4>(N, BS, iters, evStart, evStop, csv_file);
        }
        puts("");
    }

    if (csv_file) {
        fclose(csv_file);
    }

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    return 0;
}