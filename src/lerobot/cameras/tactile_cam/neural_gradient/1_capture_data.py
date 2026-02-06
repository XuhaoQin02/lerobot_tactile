"""
神经梯度标定数据采集脚本
使用 tactile_cam 的 image_processor 功能采集标定图像
"""

import cv2
import numpy as np
import os
import time
from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation


class NeuralGradientDataCapture:
    """神经梯度标定数据采集器"""
    
    def __init__(self, camera_config, save_dir="./data/calibration", save_mode="both"):
        """
        初始化数据采集器
        
        Args:
            camera_config: TactileCameraConfig 相机配置
            save_dir: 数据保存目录
            save_mode: 保存模式
                - "raw": 只保存原始图像
                - "warped": 只保存透视变换后的图像
                - "both": 同时保存原始和变换后的图像
        """
        self.camera_config = camera_config
        self.save_dir = save_dir
        self.save_mode = save_mode
        self.camera = None
        self.homography_matrix = None
        self.output_size = None
        self.calib_file = "/home/qinxuhao/lerobot/src/lerobot/cameras/tactile_cam/calibration_data/homography_matrix.npz"
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "ball_position"), exist_ok=True)
        if save_mode in ["raw", "both"]:
            os.makedirs(os.path.join(save_dir, "raw"), exist_ok=True)
        if save_mode in ["warped", "both"]:
            os.makedirs(os.path.join(save_dir, "warped"), exist_ok=True)
    
    def initialize_camera(self):
        """初始化相机"""
        print("[INFO] 初始化相机...")
        self.camera = TactileCamera(self.camera_config)
        self.camera.connect()
        print("[INFO] 相机连接成功")
    
    def load_homography(self):
        """加载透视变换矩阵"""
        try:
            calib_data = np.load(self.calib_file)
            self.homography_matrix = calib_data['homography_matrix']
            self.output_size = tuple(int(x) for x in calib_data['output_size'])
            print(f'[INFO] 成功加载透视变换矩阵，输出尺寸: {self.output_size}')
        except FileNotFoundError:
            raise FileNotFoundError(f"标定文件未找到: {self.calib_file}")
        except KeyError:
            raise KeyError("标定文件中缺少必要的键")
    
    def warp_perspective(self, image):
        """应用透视变换"""
        if self.homography_matrix is None:
            return image
        return cv2.warpPerspective(
            image, 
            self.homography_matrix, 
            self.output_size,
            flags=cv2.INTER_NEAREST
        )
    
    def save_image(self, image, filename):
        """
        根据保存模式保存图像
        
        Args:
            image: 原始BGR图像
            filename: 文件名
        """
        saved_paths = []
        
        # 保存原始图像
        if self.save_mode in ["raw", "both"]:
            raw_path = os.path.join(self.save_dir, "raw", filename)
            cv2.imwrite(raw_path, image)
            saved_paths.append(f"raw/{filename}")
        
        # 保存透视变换后的图像
        if self.save_mode in ["warped", "both"]:
            warped_img = self.warp_perspective(image)
            warped_path = os.path.join(self.save_dir, "warped", filename)
            cv2.imwrite(warped_path, warped_img)
            saved_paths.append(f"warped/{filename}")
        
        print(f'[INFO] 已保存: {", ".join(saved_paths)}')
    
    def capture_calibration_data(self, num_images=10, prefix="cal"):
        """
        采集神经梯度标定数据
        
        Args:
            num_images: 需要采集的标定图像数量
            prefix: 文件名前缀
        """
        if self.camera is None:
            raise RuntimeError("相机未初始化，请先调用 initialize_camera()")
        
        self.load_homography()
        
        print(f'[INFO] 开始采集 {num_images} 张标定图像...')
        print(f'[INFO] 保存模式: {self.save_mode}')
        print('[INFO] 操作说明:')
        print('  - 按 空格键: 捕获背景图像 (第一张，请勿按压传感器)')
        print('  - 按 空格键: 捕获标定球图像 (后续图像，按压标准球后捕获)')
        print('  - 按 m: 切换保存模式 (raw/warped/both)')
        print('  - 按 q: 退出采集')
        time.sleep(2)
        
        count = 0
        bg_captured = False
        
        while count <= num_images:
            try:
                frame = self.camera.async_read(timeout_ms=200)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                warped_img = self.warp_perspective(frame_bgr)
                
                # 显示状态信息
                display_img = warped_img.copy()
                if not bg_captured:
                    status_text = "按空格捕获背景图像 (请勿按压)"
                else:
                    status_text = f"按空格捕获标定图像 ({count}/{num_images})"
                mode_text = f"模式: {self.save_mode} (按m切换)"
                cv2.putText(display_img, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_img, mode_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.imshow('Neural Gradient Data Capture', display_img)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):
                    if not bg_captured:
                        # 保存背景图像
                        self.save_image(frame_bgr, 'bg-0.jpg')
                        bg_captured = True
                        print('[INFO] 背景图像已保存，现在可以开始采集标定球图像')
                    else:
                        # 保存标定球图像
                        self.save_image(frame_bgr, f'{prefix}-{count}.jpg')
                        count += 1
                        
                        if count > num_images:
                            print(f'[INFO] 已完成 {num_images} 张标定图像的采集')
                            break
                
                elif key == ord('m'):
                    # 切换保存模式
                    modes = ["raw", "warped", "both"]
                    current_idx = modes.index(self.save_mode)
                    self.save_mode = modes[(current_idx + 1) % len(modes)]
                    # 确保目录存在
                    if self.save_mode in ["raw", "both"]:
                        os.makedirs(os.path.join(self.save_dir, "raw"), exist_ok=True)
                    if self.save_mode in ["warped", "both"]:
                        os.makedirs(os.path.join(self.save_dir, "warped"), exist_ok=True)
                    print(f'[INFO] 切换保存模式为: {self.save_mode}')
                
                elif key == ord('q'):
                    print('[INFO] 用户取消采集')
                    break
                    
            except TimeoutError as e:
                print(f'[WARNING] 帧读取超时: {e}')
                continue
            except RuntimeError as e:
                print(f'[WARNING] 帧读取错误: {e}')
                continue
        
        cv2.destroyAllWindows()
        print(f'[INFO] 采集完成，共保存 {count} 张标定图像')
        print(f'[INFO] 数据保存在: {self.save_dir}')
        print('[INFO] 下一步: 运行标定球位置标注程序，标注球心位置')
    
    def cleanup(self):
        """清理资源"""
        if self.camera is not None:
            self.camera.disconnect()
            print("[INFO] 相机已断开连接")


def main():
    # 相机配置
    camera_config = TactileCameraConfig(
        index_or_path="/dev/video0", 
        fps=25,                       
        width=640,                   
        height=480,
        color_mode=ColorMode.RGB,     
        rotation=Cv2Rotation.NO_ROTATION, 
        exposure=600,
        wb_temperature=4500,
        r_gain=1.0,
        g_gain=1.0,
        b_gain=1.0
    )
    
    # 数据保存目录
    save_dir = os.path.join(os.path.dirname(__file__), "data/calibration")
    
    # 保存模式: "raw" / "warped" / "both"
    save_mode = "both"
    
    capture = NeuralGradientDataCapture(camera_config, save_dir, save_mode)
    
    try:
        capture.initialize_camera()
        capture.capture_calibration_data(num_images=10, prefix="cal")
        
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断采集")
    except Exception as e:
        print(f'[ERROR] 采集过程中发生错误: {e}')
        raise
    finally:
        capture.cleanup()


if __name__ == "__main__":
    main()