# check_imports.py
print("--- 开始导入诊断 ---")

try:
    print("--> 正在尝试导入 'tqdm'...")
    from tqdm import tqdm
    print("    'tqdm' 导入成功！")

    print("--> 正在尝试导入 'core.config'...")
    from core.config import cfg, args
    print("    'core.config' 导入成功！")

    print("--> 正在尝试导入 'core.utils'...")
    import core.utils as utils
    print("    'core.utils' 导入成功！")

    print("--> 正在尝试导入 'tensorflow'...")
    import tensorflow as tf
    print("    'tensorflow' 导入成功！")

    print("--> 正在尝试导入 'cv2' (OpenCV)...")
    import cv2
    print("    'cv2' (OpenCV) 导入成功！")

    print("--> 正在尝试导入 'core.dataset'...")
    import core.dataset as dataset
    print("    'core.dataset' 导入成功！")

    print("\n--- ✅ 所有核心库均已成功导入！ ---")

except Exception as e:
    print(f"\n--- ❌ 在导入过程中发生错误！ ---")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")