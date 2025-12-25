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
#include "PaDiMDetector.h"

namespace fs = std::filesystem;

// ==========================================
// 1. 线程安全的图片缓冲区
// ==========================================

struct FrameData {
    cv::Mat img;
    std::string name;
};

class ImageBuffer {
public:
    ImageBuffer(size_t maxSize) : max_size(maxSize), stop_flag(false) {}

    // 生产者调用：放入图片
    void push(const FrameData& data) {
        std::unique_lock<std::mutex> lock(mtx);
        // 如果队列满了，生产者等待 (防止内存爆满)
        not_full.wait(lock, [this] { return queue.size() < max_size || stop_flag; });
        
        if (stop_flag) return;

        queue.push(data);
        lock.unlock();
        not_empty.notify_one(); // 通知消费者有货了
    }

    // 消费者调用：取出图片
    // 返回 false 表示队列已空且生产者已停止（任务结束）
    bool pop(FrameData& data) {
        std::unique_lock<std::mutex> lock(mtx);
        // 如果队列空了，消费者等待
        not_empty.wait(lock, [this] { return !queue.empty() || stop_flag; });

        if (queue.empty() && stop_flag) {
            return false; // 任务结束
        }

        data = queue.front();
        queue.pop();
        lock.unlock();
        not_full.notify_one(); // 通知生产者可以继续放了
        return true;
    }

    // 标记生产结束
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
    // 1. 定义命令行参数规则
    // 格式: "{ 长参数名 短参数名 | 默认值 | 说明 }"
    const std::string keys =
        "{help h usage ? |      | 显示帮助信息 }"
        "{model_dir m    |      | [必须] 模型文件夹路径 (包含 onnx 和 params) }"
        "{input_dir i    |      | [必须] 测试图片文件夹路径 }"
        "{output_dir o   |      | [必须] 结果保存路径 }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("PaDiM C++ 推理程序 v1.0");

    // 如果用户输入了 -h 或 --help，打印帮助并退出
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    // 2. 获取参数值
    std::string modelDir = parser.get<std::string>("model_dir");
    std::string testDir = parser.get<std::string>("input_dir");
    std::string outputDir = parser.get<std::string>("output_dir");

    // 3. 强制检查：如果参数为空，则报错
    if (!parser.check() || modelDir.empty() || testDir.empty() || outputDir.empty()) {
        std::cerr << "❌ 错误: 缺少必要参数！" << std::endl;
        std::cerr << "请务必指定 --model_dir, --input_dir 和 --output_dir" << std::endl;
        parser.printMessage(); 
        return -1;
    }

    // 打印当前配置
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "📂 模型路径: " << modelDir << std::endl;
    std::cout << "📂 输入路径: " << testDir << std::endl;
    std::cout << "📂 输出路径: " << outputDir << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    std::cout << "🚀 正在初始化 PaDiM 检测器..." << std::endl;
    PaDiMDetector detector(modelDir);

    if (!fs::exists(outputDir)) fs::create_directory(outputDir);
    if (!fs::exists(testDir)) {
        std::cerr << "❌ 输入文件夹不存在: " << testDir << std::endl;
        return -1;
    }

    // 收集路径
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

    // 🔥 热身
    std::cout << "🔥 正在热身 GPU..." << std::endl;
    cv::Mat dummy = cv::Mat::zeros(112, 112, CV_8UC3);
    detector.predict(dummy); 
    std::cout << "✅ 热身完成！" << std::endl;

    // ==========================================
    // 🚀 启动生产者-消费者模型
    // ==========================================
    
    // 创建缓冲区，最大容量 50 张 (防止内存溢出，同时足够缓冲 IO 波动)
    ImageBuffer buffer(50); 

    // --- 启动生产者线程 (负责读图) ---
    std::thread producerThread([&]() {
        std::cout << "🧵 [生产者] 线程启动，开始读取 " << imagePaths.size() << " 张图片..." << std::endl;
        for (const auto& path : imagePaths) {
            cv::Mat img = cv::imread(path.string());
            if (!img.empty()) {
                // 将图片推入缓冲区 (如果满了会阻塞在这里等待)
                buffer.push({img, path.filename().string()});
            }
        }
        // 读完所有图片，标记结束
        buffer.stop();
        std::cout << "🧵 [生产者] 图片读取完毕，线程退出。" << std::endl;
    });

    // --- 消费者逻辑 (主线程负责推理) ---
    std::cout << "🚀 [消费者] 开始推理循环..." << std::endl;

    int count = 0;
    double totalTime = 0;
    const float anomaly_threshold = 22.0f;
    FrameData currentFrame;

    // 只要能从缓冲区取到数据，就一直循环
    while (buffer.pop(currentFrame)) {
        count++;
        // std::cout << "[" << count << "] 处理: " << currentFrame.name << "... ";

        auto start = std::chrono::high_resolution_clock::now();
        
        // 推理
        auto result = detector.predict(currentFrame.img);
        
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::milli>(end - start).count();
        totalTime += time;
        
        float anomaly_score = result.second;
        std::string status = (anomaly_score > anomaly_threshold) ? "异常" : "正常";

        std::cout << "[" << count << "] " << currentFrame.name 
                  << " | 得分: " << anomaly_score 
                  << " | 状态: " << status 
                  << " | 耗时: " << time << " ms" << std::endl;

        // ⚠️ 注意：为了不阻塞消费者线程，建议仅在异常时保存，或者将保存任务也放入另一个线程
        // 这里为了演示修复了之前的 imwrite 警告问题
        if (anomaly_score > anomaly_threshold) {
            cv::Mat amap_norm, amap_color;
            cv::normalize(result.first, amap_norm, 0, 255, cv::NORM_MINMAX);
            amap_norm.convertTo(amap_norm, CV_8U);
            cv::applyColorMap(amap_norm, amap_color, cv::COLORMAP_JET);
            cv::imwrite(outputDir + "/" + currentFrame.name, amap_color);
        }
    }

    // 等待生产者线程彻底结束
    if (producerThread.joinable()) {
        producerThread.join();
    }

    if (count > 0) {
        std::cout << "\n--------------------------------" << std::endl;
        std::cout << "✅ 处理完成！共 " << count << " 张图片" << std::endl;
        std::cout << "⚡ 平均推理耗时: " << totalTime / count << " ms/张" << std::endl;
        std::cout << "🚀 平均 FPS: " << 1000.0 / (totalTime / count) << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }

    system("pause");
    return 0;
}