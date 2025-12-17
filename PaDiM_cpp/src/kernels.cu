#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdio.h>

#define MAX_DIM 1024 

// ==========================================
// 1. 预处理核函数
// 修正：必须除以 255.0，因为 PyTorch 模型期望输入是 0-1
// ==========================================
__global__ void preprocess_kernel_fused(
    const unsigned char* __restrict__ src, 
    float* __restrict__ dst,
    int srcH, int srcW, 
    int dstH, int dstW
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) return;

    // ---------------------------------------------------------
    // 🔥 双线性插值逻辑开始
    // ---------------------------------------------------------
    
    // 1. 计算缩放比例
    float scale_x = (float)srcW / dstW;
    float scale_y = (float)srcH / dstH;

    // 2. 映射回源图像坐标 (浮点数)，使用中心对齐 (Center Alignment)
    // 这种方式与 PyTorch 的 Resize(align_corners=False) 和 OpenCV 保持一致
    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    // 3. 边界限制 (防止越界)
    src_x = fmaxf(0.0f, fminf(src_x, (float)srcW - 1.00001f));
    src_y = fmaxf(0.0f, fminf(src_y, (float)srcH - 1.00001f));

    // 4. 获取四个邻近整数坐标
    // (x0, y0) 是左上角
    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // 再次确保不超过边界
    if (x1 >= srcW) x1 = srcW - 1;
    if (y1 >= srcH) y1 = srcH - 1;

    // 5. 计算权重 (距离越近权重越大)
    float dx = src_x - x0;
    float dy = src_y - y0;
    float w00 = (1.0f - dx) * (1.0f - dy); // 左上权重
    float w10 = dx * (1.0f - dy);          // 右上权重
    float w01 = (1.0f - dx) * dy;          // 左下权重
    float w11 = dx * dy;                   // 右下权重

    // 6. 读取四个像素点的值 (BGR 格式)
    // 计算四个点的内存偏移量
    int idx00 = (y0 * srcW + x0) * 3;
    int idx10 = (y0 * srcW + x1) * 3;
    int idx01 = (y1 * srcW + x0) * 3;
    int idx11 = (y1 * srcW + x1) * 3;

    // 读取 B 通道
    float b_val = w00 * src[idx00 + 0] + w10 * src[idx10 + 0] + 
                  w01 * src[idx01 + 0] + w11 * src[idx11 + 0];
    
    // 读取 G 通道
    float g_val = w00 * src[idx00 + 1] + w10 * src[idx10 + 1] + 
                  w01 * src[idx01 + 1] + w11 * src[idx11 + 1];

    // 读取 R 通道
    float r_val = w00 * src[idx00 + 2] + w10 * src[idx10 + 2] + 
                  w01 * src[idx01 + 2] + w11 * src[idx11 + 2];

    // ---------------------------------------------------------
    // 🔥 双线性插值逻辑结束
    // ---------------------------------------------------------

    // 7. 写入目标 (CHW 格式: R->G->B) 并归一化到 [0, 1]
    int area = dstH * dstW;
    int dstIdx = y * dstW + x;

    dst[dstIdx + 0 * area] = r_val / 255.0f; 
    dst[dstIdx + 1 * area] = g_val / 255.0f;
    dst[dstIdx + 2 * area] = b_val / 255.0f;
}

extern "C" void launchPreprocessFused(
    const unsigned char* src, float* dst,
    int srcH, int srcW,
    int dstH, int dstW,
    cudaStream_t stream
) {
    dim3 block(32, 32);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);
    preprocess_kernel_fused<<<grid, block, 0, stream>>>(src, dst, srcH, srcW, dstH, dstW);
}

// ==========================================
// 2. 马氏距离核函数
// 修正：既然 Python 导出了 NHWC，这里就按 NHWC 读取
// ==========================================
__global__ void mahalanobis_kernel(
    const float* __restrict__ features, // 输入: NHWC (因为 Python 做了 permute)
    const float* __restrict__ means,    // 输入: NHWC
    const float* __restrict__ inv_covs, // 输入: NHWC
    float* __restrict__ dist_map,
    int C, int total_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    // 1. 准备指针 (NHWC 模式：连续内存)
    // features + idx * C 指向当前像素的 C 个通道的起始位置
    const float* feat_ptr = features + idx * C;
    const float* mean_ptr = means + idx * C;
    const float* cov_ptr = inv_covs + idx * C * C;

    // 2. 寄存器缓存
    float diff_cache[MAX_DIM];
    if (C > MAX_DIM) return; 

    #pragma unroll
    for (int i = 0; i < C; ++i) {
        // 🔥 关键修复：直接读取，因为数据是 NHWC 排列的
        diff_cache[i] = feat_ptr[i] - mean_ptr[i];
    }

    // 3. 计算马氏距离
    float dist_sq = 0.0f;

    #pragma unroll 4
    for (int i = 0; i < C; ++i) {
        float sum_row = 0.0f;
        float diff_i = diff_cache[i];
        const float* current_cov_row = cov_ptr + i * C;

        #pragma unroll 4
        for (int j = 0; j < C; ++j) {
            sum_row += current_cov_row[j] * diff_cache[j];
        }
        dist_sq += diff_i * sum_row;
    }

    dist_map[idx] = sqrtf(max(0.0f, dist_sq));
}

extern "C" void launchMahalanobisKernel(
    const float* features, 
    const float* means, 
    const float* inv_covs, 
    float* dist_map,
    int H, int W, int C,
    cudaStream_t stream
) {
    int total_pixels = H * W;
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;

    mahalanobis_kernel<<<blocks, threads, 0, stream>>>(
        features, means, inv_covs, dist_map, C, total_pixels
    );
}