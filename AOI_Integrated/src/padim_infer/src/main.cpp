#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <iomanip> // 用于 std::fixed, std::setprecision
#include "PaDiMDetector.h"

namespace fs = std::filesystem;

// ==========================================
// 1. 线程安全的图片缓冲区 (保持不变)
// ==========================================

struct FrameData {
    cv::Mat img;
    std::string name;
};

class ImageBuffer {
public:
    ImageBuffer(size_t maxSize) : max_size(maxSize), stop_flag(false) {}

    void push(const FrameData& data) {
        std::unique_lock<std::mutex> lock(mtx);
        not_full.wait(lock, [this] { return queue.size() < max_size || stop_flag; });
        if (stop_flag) return;
        queue.push(data);
        lock.unlock();
        not_empty.notify_one(); 
    }

    bool pop(FrameData& data) {
        std::unique_lock<std::mutex> lock(mtx);
        not_empty.wait(lock, [this] { return !queue.empty() || stop_flag; });
        if (queue.empty() && stop_flag) return false; 
        data = queue.front();
        queue.pop();
        lock.unlock();
        not_full.notify_one(); 
        return true;
    }

    void stop() {
        std::unique_lock<std::mutex> lock(mtx);
        stop_flag = true;
        not_empty.notify_all();
        not_full.notify_all();
    }

private:
    std::queue<FrameData> queue;
    std::mutex mtx;
    std::condition_variable not_empty;
    std::condition_variable not_full;
    size_t max_size;
    std::atomic<bool> stop_flag;
};

// ==========================================
// 2. 辅助函数
// ==========================================

bool isImageFile(const fs::path& filePath) {
    std::string ext = filePath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

// ==========================================
// 3. 主程序
// ==========================================

int main(int argc, char* argv[]) {
    // 1. 参数解析
    const std::string keys =
        "{help h usage ? |      | 显示帮助信息 }"
        "{model_dir m    |      | [必须] 模型文件夹路径 }"
        "{input_dir i    |      | [必须] 测试图片文件夹路径 }"
        "{output_dir o   |      | [必须] 结果保存路径 }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("PaDiM C++ 推理程序 v2.0 (智能复判版)");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string modelDir = parser.get<std::string>("model_dir");
    std::string testDir = parser.get<std::string>("input_dir");
    std::string outputDir = parser.get<std::string>("output_dir");

    if (!parser.check() || modelDir.empty() || testDir.empty() || outputDir.empty()) {
        std::cerr << "❌ 错误: 缺少必要参数！" << std::endl;
        parser.printMessage(); return -1;
    }

    // 2. 初始化
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "📂 模型路径: " << modelDir << std::endl;
    std::cout << "📂 输入路径: " << testDir << std::endl;
    std::cout << "📂 输出路径(待复判区): " << outputDir << std::endl; // 修改提示
    std::cout << "------------------------------------------------" << std::endl;

    PaDiMDetector detector(modelDir);

    // 确保输出目录存在
    // 这里的 outputDir 实际上就是 Dataset_Review/Pending 
    if (!fs::exists(outputDir)) fs::create_directories(outputDir);
    if (!fs::exists(testDir)) {
        std::cerr << "❌ 输入文件夹不存在: " << testDir << std::endl;
        return -1;
    }

    // 收集图片路径
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

    // 热身
    detector.warmup();

    // 3. 启动流水线
    ImageBuffer buffer(50); 

    // --- 生产者线程 ---
    std::thread producerThread([&]() {
        std::cout << "🧵 [生产者] 开始读取 " << imagePaths.size() << " 张图片..." << std::endl;
        for (const auto& path : imagePaths) {
            cv::Mat img = cv::imread(path.string());
            if (!img.empty()) {
                buffer.push({img, path.filename().string()});
            }
        }
        buffer.stop();
        std::cout << "🧵 [生产者] 读取完毕。" << std::endl;
    });

    // --- 消费者线程 (主线程) ---
    std::cout << "🚀 [消费者] 开始推理循环..." << std::endl;

    int count = 0;
    double totalTime = 0;
    
    // 设定阈值: 根据实际情况调整，因为后面有复判兜底，这里可以适当放宽一点(即调低一点)
    // 目的: 宁可报错杀过，不可漏网一个
    const float anomaly_threshold = 20.0f; 
    FrameData currentFrame;

    while (buffer.pop(currentFrame)) {
        count++;
        auto start = std::chrono::high_resolution_clock::now();
        
        // A. 推理
        auto result = detector.predict(currentFrame.img);
        
        // B. 决策
        float anomaly_score = result.second;
        bool is_suspect = anomaly_score > anomaly_threshold;

        // C. 处理 NG
        // 关键修改：不再保存热力图，而是保存原始图给 ResNet
        if (is_suspect) {
            std::string baseName = fs::path(currentFrame.name).stem().string();
            
            // 调用 dumpNGImage: 保存原图到 Pending 文件夹
            // 注意：你需要在 PaDiMDetector 中确保实现了这个函数
            detector.dumpNGImage(currentFrame.img, anomaly_score, outputDir, baseName);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::milli>(end - start).count();
        totalTime += time;
        
        // D. 控制台打印
        std::string status_icon;
        if (is_suspect) status_icon = "\033[31m❌ 待复判\033[0m"; // 红色
        else            status_icon = "\033[32m✅ OK\033[0m";     // 绿色

        std::cout << "[" << count << "] " << currentFrame.name 
                  << " | 得分: " << std::fixed << std::setprecision(2) << anomaly_score 
                  << " | 状态: " << status_icon 
                  << " | 耗时: " << std::fixed << std::setprecision(1) << time << " ms" << std::endl;
    }

    if (producerThread.joinable()) producerThread.join();

    if (count > 0) {
        std::cout << "\n--------------------------------" << std::endl;
        std::cout << "✅ 完成！共 " << count << " 张" << std::endl;
        std::cout << "⚡ 平均FPS: " << 1000.0 / (totalTime / count) << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }

    system("pause");
    return 0;
}