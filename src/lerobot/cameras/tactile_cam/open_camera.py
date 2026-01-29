"""
相机打开脚本
- 分辨率: 640x480
- 帧率: 60fps
- 关闭自动曝光
- 关闭自动白平衡
"""

import cv2
import platform
import os
import subprocess
import re

# Windows 兼容性设置
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"


def get_v4l2_exposure_range(camera_index: int = 0) -> dict:
    """
    使用 v4l2-ctl 获取相机曝光参数范围
    
    Returns:
        dict: 包含 min, max, default, step 的字典，如果获取失败则返回默认值
    """
    default_range = {"min": 1, "max": 10000, "default": 100, "step": 1}
    
    if platform.system() != "Linux":
        return default_range
    
    try:
        # 使用 v4l2-ctl 获取曝光参数信息
        result = subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{camera_index}", "-l"],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode != 0:
            print(f"v4l2-ctl 执行失败: {result.stderr}")
            return default_range
        
        output = result.stdout
        
        # 查找 exposure_absolute 或 exposure_time_absolute 参数
        exposure_patterns = [
            r"exposure_absolute.*?min=(-?\d+).*?max=(-?\d+).*?step=(\d+).*?default=(-?\d+)",
            r"exposure_time_absolute.*?min=(-?\d+).*?max=(-?\d+).*?step=(\d+).*?default=(-?\d+)",
            r"exposure.*?min=(-?\d+).*?max=(-?\d+).*?step=(\d+).*?default=(-?\d+)",
        ]
        
        for pattern in exposure_patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
            if match:
                exposure_range = {
                    "min": int(match.group(1)),
                    "max": int(match.group(2)),
                    "step": int(match.group(3)),
                    "default": int(match.group(4))
                }
                print(f"检测到曝光范围: min={exposure_range['min']}, max={exposure_range['max']}, "
                      f"step={exposure_range['step']}, default={exposure_range['default']}")
                return exposure_range
        
        print("未能从 v4l2-ctl 输出中解析曝光参数，使用默认值")
        print(f"v4l2-ctl 输出: {output[:500]}")  # 打印部分输出用于调试
        
    except FileNotFoundError:
        print("v4l2-ctl 未安装，使用默认曝光范围")
    except subprocess.TimeoutExpired:
        print("v4l2-ctl 执行超时，使用默认曝光范围")
    except Exception as e:
        print(f"获取曝光范围时出错: {e}")
    
    return default_range


def set_v4l2_exposure(camera_index: int, value: int) -> bool:
    """
    使用 v4l2-ctl 设置曝光值
    
    Args:
        camera_index: 相机索引
        value: 曝光值
    
    Returns:
        bool: 是否设置成功
    """
    if platform.system() != "Linux":
        return False
    
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{camera_index}", 
             f"--set-ctrl=exposure_time_absolute={value}"],
            capture_output=True, text=True, timeout=2
        )
        return result.returncode == 0
    except Exception as e:
        print(f"设置曝光值失败: {e}")
        return False


def set_v4l2_auto_exposure(camera_index: int, manual: bool = True) -> bool:
    """
    使用 v4l2-ctl 设置自动曝光模式
    
    Args:
        camera_index: 相机索引
        manual: True 为手动模式，False 为自动模式
    
    Returns:
        bool: 是否设置成功
    """
    if platform.system() != "Linux":
        return False
    
    try:
        # 1 = Manual Mode, 3 = Aperture Priority Mode (auto)
        mode = 1 if manual else 3
        result = subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{camera_index}", 
             f"--set-ctrl=auto_exposure={mode}"],
            capture_output=True, text=True, timeout=2
        )
        return result.returncode == 0
    except Exception as e:
        print(f"设置自动曝光模式失败: {e}")
        return False


def open_camera(camera_index: int = 0):
    """
    打开相机并配置参数
    
    Args:
        camera_index: 相机索引，默认为0
    
    Returns:
        cv2.VideoCapture: 配置好的相机对象
    """
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机 {camera_index}")
    
    # 设置分辨率: 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 设置帧率: 60fps
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # 关闭自动白平衡 (0 = 关闭)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    
    # 关闭自动曝光并设置手动曝光模式
    if platform.system() == "Linux":
        # 使用 v4l2-ctl 设置曝光，因为 OpenCV 的 CAP_PROP_EXPOSURE 在某些相机上不起作用
        exp_range = get_v4l2_exposure_range(camera_index)
        
        # 设置为手动曝光模式
        if set_v4l2_auto_exposure(camera_index, manual=True):
            print("已设置为手动曝光模式")
        else:
            # 备用方案：使用 OpenCV
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        # 设置初始曝光值为默认值
        initial_exposure = exp_range["default"]
        if set_v4l2_exposure(camera_index, initial_exposure):
            print(f"曝光值已设置为: {initial_exposure}")
        else:
            # 备用方案：使用 OpenCV
            cap.set(cv2.CAP_PROP_EXPOSURE, initial_exposure)
    else:
        # Windows 使用不同的值
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = 手动模式
        cap.set(cv2.CAP_PROP_EXPOSURE, -4)
    
    # 验证设置
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    auto_exposure = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    auto_wb = cap.get(cv2.CAP_PROP_AUTO_WB)
    
    print(f"相机配置:")
    print(f"  分辨率: {actual_width}x{actual_height}")
    print(f"  帧率: {actual_fps:.1f} fps")
    print(f"  自动曝光模式: {auto_exposure} (1=手动, 3=自动)")
    print(f"  当前曝光值: {current_exposure}")
    print(f"  自动白平衡: {auto_wb}")
    
    return cap


def apply_rgb_gain(frame, r_gain, g_gain, b_gain):
    """
    应用RGB增益到图像
    
    Args:
        frame: 输入BGR图像
        r_gain: 红色通道增益 (0.0 - 3.0)
        g_gain: 绿色通道增益 (0.0 - 3.0)
        b_gain: 蓝色通道增益 (0.0 - 3.0)
    
    Returns:
        调整后的图像
    """
    import numpy as np
    
    # 分离通道 (OpenCV是BGR顺序)
    b, g, r = cv2.split(frame.astype(np.float32))
    
    # 应用增益
    r = np.clip(r * r_gain, 0, 255)
    g = np.clip(g * g_gain, 0, 255)
    b = np.clip(b * b_gain, 0, 255)
    
    # 合并通道
    result = cv2.merge([b, g, r]).astype(np.uint8)
    return result


def main():
    """主函数：打开相机并实时显示画面"""
    print("正在打开相机...")
    cap = open_camera(camera_index=0)
    
    print("\n按 'q' 键退出")
    print("按 '+' / '-' 键调整曝光值 (亮/暗)")
    print("按 'w' / 'W' 键调整白平衡色温 (+/-)")
    print("按 'r' / 'R' 键调整红色增益 (+/-)")
    print("按 'g' / 'G' 键调整绿色增益 (+/-)")
    print("按 'b' / 'B' 键调整蓝色增益 (+/-)")
    print("按 '0' 键重置RGB增益为1.0")
    
    # 当前曝光值和白平衡值
    # Linux V4L2 使用正数曝光值，Windows 使用负数曝光值
    if platform.system() == "Linux":
        # 尝试使用 v4l2-ctl 获取实际曝光范围
        exp_range = get_v4l2_exposure_range(0)
        exposure_value = exp_range["default"]
        exposure_min = exp_range["min"]
        exposure_max = exp_range["max"]
        exposure_step = max(exp_range["step"], (exposure_max - exposure_min) // 150)  # 至少1%步进
        print(f"曝光范围: {exposure_min} - {exposure_max}, 步进: {exposure_step}, 当前值: {exposure_value}")
    else:
        exposure_value = -4   # Windows: 负数曝光值
        exposure_min = -13
        exposure_max = -1
        exposure_step = 1
    
    wb_value = 4000     # 白平衡色温，常见范围: 2000-8000
    
    # RGB增益值 (范围: 0.0 - 3.0)
    r_gain = 1.15
    g_gain = 1.0
    b_gain = 1.0
    gain_step = 0.05
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("无法读取帧")
            break
        
        # 应用RGB增益
        frame = apply_rgb_gain(frame, r_gain, g_gain, b_gain)
        
        # 在画面上显示信息
        info_text = f"640x480@60fps | Exp:{exposure_value} | WB:{wb_value}K"
        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1)
        
        rgb_text = f"R:{r_gain:.1f} G:{g_gain:.1f} B:{b_gain:.1f} | Press 'q' to quit"
        cv2.putText(frame, rgb_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1)
        
        # 显示画面
        cv2.imshow("Camera - 640x480 @ 60fps", frame)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("退出...")
            break
        elif key == ord('+') or key == ord('='):
            # 增加曝光值 (更亮) - 使用 + 或 = 键
            exposure_value = min(exposure_value + exposure_step, exposure_max)
            if platform.system() == "Linux":
                set_v4l2_exposure(0, exposure_value)
            else:
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
            print(f"曝光值设置为: {exposure_value} (更亮)")
        elif key == ord('-') or key == ord('_'):
            # 减少曝光值 (更暗) - 使用 - 或 _ 键
            exposure_value = max(exposure_value - exposure_step, exposure_min)
            if platform.system() == "Linux":
                set_v4l2_exposure(0, exposure_value)
            else:
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
            print(f"曝光值设置为: {exposure_value} (更暗)")
        elif key == ord('w'):
            # 增加白平衡色温
            wb_value = min(wb_value + 500, 8000)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_value)
            print(f"白平衡色温设置为: {wb_value}K")
        elif key == ord('W'):
            # 减少白平衡色温
            wb_value = max(wb_value - 500, 2000)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_value)
            print(f"白平衡色温设置为: {wb_value}K")
        elif key == ord('r'):
            # 增加红色增益
            r_gain = min(r_gain + gain_step, 3.0)
            print(f"红色增益: {r_gain:.1f}")
        elif key == ord('R'):
            # 减少红色增益
            r_gain = max(r_gain - gain_step, 0.0)
            print(f"红色增益: {r_gain:.1f}")
        elif key == ord('g'):
            # 增加绿色增益
            g_gain = min(g_gain + gain_step, 3.0)
            print(f"绿色增益: {g_gain:.1f}")
        elif key == ord('G'):
            # 减少绿色增益
            g_gain = max(g_gain - gain_step, 0.0)
            print(f"绿色增益: {g_gain:.1f}")
        elif key == ord('b'):
            # 增加蓝色增益
            b_gain = min(b_gain + gain_step, 3.0)
            print(f"蓝色增益: {b_gain:.1f}")
        elif key == ord('B'):
            # 减少蓝色增益
            b_gain = max(b_gain - gain_step, 0.0)
            print(f"蓝色增益: {b_gain:.1f}")
        elif key == ord('0'):
            # 重置RGB增益
            r_gain = g_gain = b_gain = 1.0
            print("RGB增益已重置为1.0")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("相机已关闭")


if __name__ == "__main__":
    main()
