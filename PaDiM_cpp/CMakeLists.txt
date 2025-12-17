cmake_minimum_required(VERSION 3.10)
project(PaDiM_cpp)


set(CMAKE_CXX_STANDARD 17)

# 屏蔽旧版 CUDA 写法的警告
cmake_policy(SET CMP0146 OLD)

# =========================================================
# 👇👇👇 1. 配置库路径 (请根据你的实际解压位置修改这里！) 👇👇👇
# =========================================================

# CUDA 路径 
# (去 D 盘看看是 D:/CUDA 还是 D:/CUDA/v11.8，要指向包含 bin/include 的那一层)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6") 

# TensorRT 路径 (指向你解压的根目录)
set(TENSORRT_ROOT "E:/Code/PaDiM_cpp/lib/TensorRT-8.6.1.6")

# OpenCV 路径 (指向 build 目录)
set(OpenCV_DIR "E:/Code/PaDiM_cpp/lib/opencv/build") 

# =========================================================
# 👆👆👆 修改上面即可，下面通常不需要动 👆👆👆
# =========================================================

# 解决 MSVC 运行时库冲突 (关键修复)
if(MSVC)
    # 统一使用多线程动态库 (/MD Release, /MDd Debug)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    # 忽略 LIBCMT 库以解决冲突
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT.lib")
endif()

# 2. 查找包
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# 3. 包含头文件
include_directories(
    "${CUDA_TOOLKIT_ROOT_DIR}/include"  # 👈 必须添加这一行！
    ${CUDA_INCLUDE_DIRS}
    "${TENSORRT_ROOT}/include"
    ${OpenCV_INCLUDE_DIRS}
    "src"
)

# 4. 链接库目录
link_directories(
    "${TENSORRT_ROOT}/lib"
)

# 5. 收集源文件 (自动查找 src 下所有的 .cpp 和 .cu)
file(GLOB_RECURSE SRCS "src/*.cpp" "src/*.cu")

# 6. 生成可执行文件 (使用 cuda_add_executable 处理 .cu 文件)
cuda_add_executable(padim_infer ${SRCS})

# 7. 链接依赖库
target_link_libraries(padim_infer 
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES} 
    nvinfer           # TensorRT 核心库
    nvonnxparser      # ONNX 解析库
)
