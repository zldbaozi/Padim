#include "PaDiMDetector.h"
#include <NvOnnxParser.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

// 引入 CUDA 函数声明 (与 .h 保持一致)
// 注意：这里不需要重复声明，因为已经 include 了 PaDiMDetector.h
// 但为了保险起见，保留 extern "C" 声明也是可以的，或者直接依赖头文件

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

void check(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << " \"" << func << "\" \n";
        exit(EXIT_FAILURE);
    }
}

PaDiMDetector::PaDiMDetector(const std::string& modelDir) {
    // 创建 CUDA 流
    checkCudaErrors(cudaStreamCreate(&stream));
    
    d_raw_image = nullptr;
    d_raw_image_size = 0;

    std::string configPath = modelDir + "/config.txt";
    std::string onnxPath = modelDir + "/padim_backbone.onnx";
    std::string meansPath = modelDir + "/means.bin";
    std::string covsPath = modelDir + "/inv_covs.bin";

    loadConfig(configPath);
    // 1. 新增：在加载引擎前，先分配好固定缓冲区
    // 此时 input_w/h 等参数已经通过 loadConfig 读取完毕
    allocateFixedBuffers();
    loadEngine(onnxPath);
    loadParams(meansPath, covsPath);

    //🔥 新增：构造函数最后调用热身
    warmup();
}

PaDiMDetector::~PaDiMDetector() {
    // 释放显存
    if (d_raw_image) cudaFree(d_raw_image);
    if (d_input) cudaFree(d_input);
    if (d_features) cudaFree(d_features);
    if (d_means) cudaFree(d_means);
    if (d_inv_covs) cudaFree(d_inv_covs);
    if (d_dist_map) cudaFree(d_dist_map);
    

    // 释放 TensorRT 资源
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;

    // 销毁流
    if (stream) cudaStreamDestroy(stream);
}

void PaDiMDetector::loadConfig(const std::string& configPath) {
    std::cout << "📖 读取配置文件: " << configPath << std::endl;
    std::ifstream file(configPath);
    if (!file.is_open()) {
        std::cerr << "⚠️ 无法打开配置文件，将使用默认参数 (64x64)" << std::endl;
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto delimiterPos = line.find("=");
        if (delimiterPos == std::string::npos) continue;

        std::string key = line.substr(0, delimiterPos);
        std::string value = line.substr(delimiterPos + 1);

        key.erase(0, key.find_first_not_of(" \t\r\n"));
        key.erase(key.find_last_not_of(" \t\r\n") + 1);
        value.erase(0, value.find_first_not_of(" \t\r\n"));
        value.erase(value.find_last_not_of(" \t\r\n") + 1);

        try {
            int val = std::stoi(value);
            if (key == "input_width") input_w = val;
            else if (key == "input_height") input_h = val;
            else if (key == "feature_map_w") feat_w = val;
            else if (key == "feature_map_h") feat_h = val;
            else if (key == "feature_dim") feat_c = val;
        } catch (...) {
            continue;
        }
    }

     std::cout << "   ✅ 配置已加载: Input=" << input_w << "x" << input_h 
              << ", Feat=" << feat_w << "x" << feat_h << ", Dim=" << feat_c << std::endl;
}

// 新增：实现固定缓冲区分配
void PaDiMDetector::allocateFixedBuffers() {
    std::cout << "💾 分配固定 GPU 缓冲区..." << std::endl;

    // 1. TensorRT 输入 (Batch=1, CHW)
    size_t input_size = 1 * 3 * input_h * input_w * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_input, input_size));

    // 2. TensorRT 输出特征
    size_t feat_size = 1 * feat_h * feat_w * feat_c * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_features, feat_size));

    // 3. 距离图 (原始 & 模糊)
    size_t map_size = feat_h * feat_w * sizeof(float);
    checkCudaErrors(cudaMalloc(&d_dist_map, map_size));

    // 4. 🔥 优化：预分配原始图像缓冲区 (假设最大 4K 分辨率，避免 predict 中动态分配)
    // 3840 * 2160 * 3 = ~24MB，对于显存来说很小
    size_t max_image_size = 3840 * 2160 * 3 * sizeof(unsigned char);
    checkCudaErrors(cudaMalloc((void**)&d_raw_image, max_image_size));
    d_raw_image_size = max_image_size;

    std::cout << "   - Input Buffer: " << input_size / 1024 << " KB" << std::endl;
    std::cout << "   - Feature Buffer: " << feat_size / 1024 << " KB" << std::endl;
    std::cout << "   - Raw Image Buffer (Max): " << max_image_size / 1024 / 1024 << " MB" << std::endl;
}

// 🔥 新增：热身函数
void PaDiMDetector::warmup() {
    std::cout << "🔥 正在热身 GPU (锁定频率)..." << std::endl;
    // 创建一个假的黑色图像
    cv::Mat dummy = cv::Mat::zeros(input_h, input_w, CV_8UC3);
    
    // 连续推理 10 次，强制 GPU 升频并保持状态
    for(int i=0; i<10; ++i) {
        predict(dummy);
    }
    // 强制同步，确保热身完成
    cudaStreamSynchronize(stream);
    std::cout << "✅ 热身完成，推理引擎已就绪" << std::endl;
}


void PaDiMDetector::loadEngine(const std::string& onnxPath) {
    std::cout << "🔄 构建 TensorRT 引擎: " << onnxPath << std::endl;
    
    runtime = nvinfer1::createInferRuntime(gLogger);
    auto builder = nvinfer1::createInferBuilder(gLogger);
    auto network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto parser = nvonnxparser::createParser(*network, gLogger);

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "❌ ONNX 解析失败" << std::endl;
        exit(1);
    }

    auto config = builder->createBuilderConfig();
    auto profile = builder->createOptimizationProfile();
    int nbInputs = network->getNbInputs();
    
    for (int i = 0; i < nbInputs; i++) {
        auto input = network->getInput(i);
        const char* name = input->getName();
        
        nvinfer1::Dims4 inputDims;
        inputDims.nbDims = 4;
        inputDims.d[0] = 1;       // Batch
        inputDims.d[1] = 3;       // Channels
        inputDims.d[2] = input_h; // Height
        inputDims.d[3] = input_w; // Width

        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, inputDims);
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, inputDims);
        profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, inputDims);
    }
    config->addOptimizationProfile(profile);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);
    
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "⚡ FP16 加速已开启" << std::endl;
    }

    auto plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) {
        std::cerr << "❌ 引擎构建失败" << std::endl;
        exit(1);
    }

    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    context = engine->createExecutionContext();

    nvinfer1::Dims4 runDims;
    runDims.nbDims = 4;
    runDims.d[0] = 1; runDims.d[1] = 3; runDims.d[2] = input_h; runDims.d[3] = input_w;
    context->setBindingDimensions(0, runDims);

    delete plan;
    delete config;
    delete parser;
    delete network;
    delete builder;
}

void PaDiMDetector::loadParams(const std::string& meansPath, const std::string& covsPath) {
    size_t num_pixels = feat_h * feat_w;
    size_t means_size = num_pixels * feat_c * sizeof(float);
    size_t covs_size = num_pixels * feat_c * feat_c * sizeof(float);

    std::vector<char> means_buf(means_size);
    std::vector<char> covs_buf(covs_size);

    std::ifstream f_means(meansPath, std::ios::binary);
    if (!f_means) { std::cerr << "❌ 无法读取: " << meansPath << std::endl; exit(1); }
    f_means.read(means_buf.data(), means_size);
    
    std::ifstream f_covs(covsPath, std::ios::binary);
    if (!f_covs) { std::cerr << "❌ 无法读取: " << covsPath << std::endl; exit(1); }
    f_covs.read(covs_buf.data(), covs_size);

    checkCudaErrors(cudaMalloc(&d_means, means_size));
    checkCudaErrors(cudaMalloc(&d_inv_covs, covs_size));

    checkCudaErrors(cudaMemcpyAsync(d_means, means_buf.data(), means_size, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_inv_covs, covs_buf.data(), covs_size, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    
    std::cout << "✅ 统计参数已加载到 GPU" << std::endl;
}

std::pair<cv::Mat, float> PaDiMDetector::predict(const cv::Mat& img) {
    // 1. (移除懒加载，直接加载图片)
    size_t imgSize = img.rows * img.cols * 3 * sizeof(unsigned char);

    // 🔥 优化：移除动态扩容逻辑
    // 我们已经在 allocateFixedBuffers 中分配了足够大的空间
    // 如果图片真的超过 4K，这里加一个安全检查即可
    if (imgSize > d_raw_image_size) {
        std::cerr << "❌ 错误: 输入图片过大，超过预分配缓冲区!" << std::endl;
        return {cv::Mat(), 0.0f};
    }
    
    // =========================================================
    // 🚀 异步发射阶段 (CPU 几乎瞬间完成，不等待 GPU)
    // =========================================================

    // 2. 上传
    checkCudaErrors(cudaMemcpyAsync(d_raw_image, img.data, imgSize, cudaMemcpyHostToDevice, stream));

    // 3. 预处理
    launchPreprocessFused(
        d_raw_image, (float*)d_input,
        img.rows, img.cols, 
        input_h, input_w,   
        stream
    );

    // 4. 推理
    void* bindings[] = {d_input, d_features};
    context->enqueueV2(bindings, stream, nullptr);

    // 5. 马氏距离
    launchMahalanobisKernel(
        (float*)d_features, (float*)d_means, (float*)d_inv_covs, (float*)d_dist_map,
        feat_h, feat_w, feat_c, stream
    );

    // 6. 下载结果 (只下载 28x28 的小图，速度极快)
    cv::Mat amap(feat_h, feat_w, CV_32FC1);
    checkCudaErrors(cudaMemcpyAsync(amap.data, d_dist_map, feat_h * feat_w * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // 🛑 同步点
    checkCudaErrors(cudaStreamSynchronize(stream));

    // =========================================================
    // 🖥️ CPU 后处理
    // =========================================================
    
    double minVal, maxVal;
    cv::minMaxLoc(amap, &minVal, &maxVal);

    // ❌ 不需要 Resize 了，直接用小图计算得分
    // cv::Mat result_map;
    // cv::resize(amap, result_map, img.size()); 

    float max_threshold = 100.0f; 
    float score_percentage = (float)maxVal / max_threshold * 100.0f;
    if (score_percentage > 100.0f) score_percentage = 100.0f;

    // 返回小图 amap 即可
    return {amap, score_percentage}; 
}
