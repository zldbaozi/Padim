// filepath: [main.cpp](http://_vscodecontentref_/1)
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "PaDiMDetector.h"

namespace fs = std::filesystem;

bool isImageFile(const fs::path& filePath) {
    std::string ext = filePath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

int main() {
    std::string modelDir = "../cpp_model_testdata2"; 
    std::string testDir = "C:\\Users\\mento\\Desktop\\data2\\NG"; 
    std::string outputDir = "./results";

    std::cout << "🚀 正在初始化 PaDiM 检测器..." << std::endl;
    PaDiMDetector detector(modelDir);

    if (!fs::exists(outputDir)) fs::create_directory(outputDir);
    if (!fs::exists(testDir)) {
        std::cerr << "❌ 文件夹不存在" << std::endl;
        return -1;
    }

    // 1. 收集所有图片路径
    std::vector<fs::path> imagePaths;
    for (const auto& entry : fs::directory_iterator(testDir)) {
        if (entry.is_regular_file() && isImageFile(entry.path())) {
            imagePaths.push_back(entry.path());
        }
    }

    if (imagePaths.empty()) {
        std::cout << "⚠️ 没有找到图片" << std::endl;
        return 0;
    }

    std::cout << "📂 找到 " << imagePaths.size() << " 张图片，开始流水线处理..." << std::endl;

    // 🔥 热身 (Warm-up): 跑一张空图，消除第一次推理的延迟
    std::cout << "🔥 正在热身 GPU..." << std::endl;
    cv::Mat dummy = cv::Mat::zeros(112, 112, CV_8UC3); // 假设输入是 112x112
    detector.predict(dummy); 
    std::cout << "✅ 热身完成！" << std::endl;

    // 2. 流水线循环
    // 策略: 预读下一张 (Next)，同时推理当前张 (Current)
    
    cv::Mat currImg, nextImg;
    std::string currName, nextName;

    // 预读第一张
    nextImg = cv::imread(imagePaths[0].string());
    nextName = imagePaths[0].filename().string();

    int count = 0;
    double totalTime = 0;
    
    // 🔥 设置异常分数阈值
    const float anomaly_threshold = 9.0f; // 设定异常分数阈值

    for (size_t i = 0; i < imagePaths.size(); ++i) {
        // 移交所有权: Next -> Current
        currImg = nextImg;
        currName = nextName;

        // 🚀 关键优化: 在 GPU 推理当前张的同时，CPU 读取下一张
        // 这样 CPU 的 IO 时间就被 GPU 的计算时间掩盖了 (Hiding Latency)
        if (i + 1 < imagePaths.size()) {
            nextName = imagePaths[i + 1].filename().string();
            // 这是一个耗时操作 (~3-5ms)，现在它和 detector.predict 并行了(宏观上)
            // 注意: 真正的并行需要多线程，但这里利用了 OS 的文件缓存和 GPU 异步特性
            // 为了简单起见，我们先串行读，但因为 GPU 是异步的，
            // 如果 detector.predict 内部没有强制同步 (cudaStreamSynchronize)，
            // 那么 GPU 会在后台跑，CPU 就会立刻回来读图。
            // *注意*: 你现在的 predict 里加了计时用的 Synchronize，所以这里还是串行的。
            // *如果要并行*: 必须去掉 predict 里的计时同步代码！
            nextImg = cv::imread(imagePaths[i + 1].string());
        }

        std::cout << "[" << ++count << "] 处理: " << currName << "... ";

        if (currImg.empty()) continue;

        auto start = std::chrono::high_resolution_clock::now();
        
        // 推理
        auto result = detector.predict(currImg);
        
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::milli>(end - start).count();
        totalTime += time;
        
         // 获取异常得分
        float anomaly_score = result.second;

        // 判断是否异常
        std::string status = (anomaly_score > anomaly_threshold) ? "异常" : "正常";

        // 打印结果
        std::cout << "异常得分: " << anomaly_score << " | 状态: " << status << " | 耗时: " << time << " ms" << std::endl;

        // 保存结果 (可选，也会耗时)
        cv::imwrite(outputDir + "/" + currName, result.first);
    }

   

    if (count > 0) {
        std::cout << "\n--------------------------------" << std::endl;
        std::cout << "✅ 处理完成！共 " << count << " 张图片" << std::endl;
        std::cout << "⚡ 平均耗时: " << totalTime / count << " ms/张" << std::endl;
        std::cout << "🚀 平均 FPS: " << 1000.0 / (totalTime / count) << std::endl;
        std::cout << "--------------------------------" << std::endl;
    } else {
        std::cout << "⚠️ 该文件夹下没有找到图片。" << std::endl;
    }

    system("pause");
    return 0;
}