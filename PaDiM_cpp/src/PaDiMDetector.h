#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

// CUDA 错误检查宏
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__ )
void check(cudaError_t result, char const *const func, const char *const file, int const line);

// =========================================================
// 🔥 核心修复：确保这里的声明与 kernels.cu 完全一致
// =========================================================
extern "C" void launchPreprocessFused(
    const unsigned char* src, float* dst,
    int srcH, int srcW,
    int dstH, int dstW,
    cudaStream_t stream
);

extern "C" void launchMahalanobisKernel(
    const float* features, 
    const float* means, 
    const float* inv_covs, 
    float* dist_map,
    int H, int W, int C,
    cudaStream_t stream
);

class PaDiMDetector {
public:
    PaDiMDetector(const std::string& modelDir);
    ~PaDiMDetector();

    // 推理函数：返回 {热力图, 最大异常得分}
    std::pair<cv::Mat, float> predict(const cv::Mat& img);

private:
    void loadConfig(const std::string& configPath);
    void loadEngine(const std::string& onnxPath);
    void loadParams(const std::string& meansPath, const std::string& covsPath);
    // 新增：专门分配固定大小的缓冲区 (d_input, d_features, d_dist_map)
    void allocateFixedBuffers();
    void warmup();  // 新增：热身函数

    // TensorRT 组件
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    // CUDA 流
    cudaStream_t stream = nullptr;

    unsigned char* d_raw_image = nullptr; // 存放原始大图 (预分配固定大小，最大 4K 分辨率)
    size_t d_raw_image_size = 0;          // 记录当前 d_raw_image 的容量
    
    
    void* d_input = nullptr;       // TensorRT 输入 (float, CHW, Normalized)
    void* d_features = nullptr;    // TensorRT 输出特征
    void* d_means = nullptr;       // 均值向量
    void* d_inv_covs = nullptr;    // 逆协方差矩阵
    void* d_dist_map = nullptr;    // 原始距离图
    
    // 模型参数 (从 config.txt 读取)
    int input_w = 112;
    int input_h = 112;
    int feat_w = 28;
    int feat_h = 28;
    int feat_c = 100;
};