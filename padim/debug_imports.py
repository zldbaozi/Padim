import sys
import os

def debug_imports():
    """调试所有模块导入"""
    print("=" * 50)
    print("模块导入调试工具")
    print("=" * 50)
    
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root)
    print(f"项目根目录: {project_root}")
    print()
    
    # 测试导入各个模块
    modules_to_test = [
        'models.feature_extractor',
        'models.padim_trainer', 
        'models.padim_detector',
        'utils.data_loader',
        'utils.visualization'
    ]
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"✅ {module_name}: 导入成功")
            
            # 如果是models模块，检查类是否存在
            if module_name.startswith('models.'):
                if hasattr(module, 'FeatureExtractor'):
                    print(f"   ✅ FeatureExtractor类存在")
                if hasattr(module, 'PaDiMTrainer'):
                    print(f"   ✅ PaDiMTrainer类存在")
                if hasattr(module, 'PaDiMDetector'):
                    print(f"   ✅ PaDiMDetector类存在")
                    
        except ImportError as e:
            print(f"❌ {module_name}: 导入失败 - {e}")
        except Exception as e:
            print(f"⚠️  {module_name}: 其他错误 - {e}")
    
    print()
    print("=" * 50)
    print("Python路径:")
    for path in sys.path:
        print(f"  {path}")

if __name__ == "__main__":
    debug_imports()