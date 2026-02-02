"""
基于梯度的触觉传感器测试脚本
使用 pose.py 中的 img2grad 方法计算梯度和深度，不使用查找表
"""

from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
import cv2
import numpy as np
import os
import math
import scipy.fftpack


def trim(img):
    """将图像像素值裁剪到 [0, 255]"""
    img[img < 0] = 0
    img[img > 255] = 255
    return img


def sigmoid(x):
    """Sigmoid 函数"""
    return np.exp(x) / (1 + np.exp(x))


def recon_py(img, frame0, x_ratio=0.5, y_ratio=0.5, bias=1.0):
    """
    深度重建函数（来自用户提供的实现）
    
    Args:
        img: 当前帧
        frame0: 参考帧
        x_ratio: x方向比例系数
        y_ratio: y方向比例系数
        bias: 偏置系数
        
    Returns:
        result: 重建的深度图
        dx_display: 用于显示的梯度（未缩放）
        dy_display: 用于显示的梯度（未缩放）
    """
    img = np.int32(img)
    frame0 = np.int32(frame0)
    diff = img - frame0
    diff = diff * bias

    # 计算梯度
    dx1 = diff[:, :, 1] * x_ratio / 255.0
    dy1 = (diff[:, :, 2] * y_ratio - diff[:, :, 0] * (1 - y_ratio)) / 255.0
    
    # 非线性变换（避免除零）
    dx1 = np.clip(dx1, -0.99, 0.99)
    dy1 = np.clip(dy1, -0.99, 0.99)
    
    # 非线性变换后的梯度（用于显示法向量）
    dx_display = dx1 / (1 - dx1 ** 2) ** 0.5
    dy_display = dy1 / (1 - dy1 ** 2) ** 0.5
    
    # 深度重建用缩放后的梯度
    dx = dx_display / 32
    dy = dy_display / 32

    # 计算二阶导数
    gxx = dx[:-1, 1:] - dx[:-1, :-1]
    gyy = dy[1:, :-1] - dy[:-1, :-1]

    f = np.zeros(dx.shape)
    f[:-1, 1:] += gxx
    f[1:, :-1] += gyy

    # DST 变换求解泊松方程
    tt = scipy.fftpack.dst(f, norm='ortho')
    fsin = scipy.fftpack.dst(tt.T, norm='ortho').T

    (x, y) = np.meshgrid(range(1, f.shape[1]+1), range(1, f.shape[0]+1), copy=True)
    denom = (2*np.cos(math.pi*x/(f.shape[1]+2))-2) + (2*np.cos(math.pi*y/(f.shape[0]+2)) - 2)

    f = fsin/denom

    tt = scipy.fftpack.idst(f, norm='ortho')
    img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T

    result = np.zeros(f.shape)
    result[1:-1, 1:-1] = img_tt[1:-1, 1:-1]

    # 返回未缩放的梯度用于法向量显示
    return result, dx_display, dy_display


class GradientProcessor:
    """
    基于梯度的 GelSight 图像处理器
    使用 pose.py 中的梯度计算方法，不依赖查找表
    """
    
    def __init__(self, pad=20, sensor_id="right"):
        """
        初始化处理器
        
        Args:
            pad: 边缘裁剪像素数
            sensor_id: 传感器ID，"left" 或 "right"，影响梯度计算方向
        """
        self.pad = pad
        self.sensor_id = sensor_id
        
        self.ref_frame = None  # 参考帧（第一帧）
        self.ref_blur = None   # 模糊后的参考帧
        self.con_flag = True   # 是否是第一帧
        
        self.homography_matrix = None
        self.output_size = None
        self.calib_file = "/home/qinxuhao/lerobot/src/lerobot/cameras/tactile_cam/calibration_data/homography_matrix.npz"
        
        # 加载透视变换矩阵
        self._load_homography()
    
    def _load_homography(self):
        """加载透视变换矩阵"""
        try:
            calib_data = np.load(self.calib_file)
            self.homography_matrix = calib_data['homography_matrix']
            self.output_size = tuple(int(x) for x in calib_data['output_size'])
            print(f'[INFO] 成功加载透视变换矩阵，输出尺寸: {self.output_size}')
        except Exception as e:
            print(f'[WARNING] 加载透视变换矩阵失败: {e}')
            self.homography_matrix = None
    
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
    
    def _crop_image(self, img):
        """裁剪图像边缘"""
        if self.pad > 0:
            return img[self.pad:-self.pad, self.pad:-self.pad]
        return img
    
    def img2grad(self, frame0, frame, bias=4.0):
        """
        从图像差异计算梯度
        
        Args:
            frame0: 参考帧（无接触时）
            frame: 当前帧
            bias: 偏置系数
            
        Returns:
            dx, dy: x和y方向的梯度
        """
        diff = frame.astype(np.float32) - frame0.astype(np.float32)
        diff = diff * bias
        
        # OpenCV BGR 顺序: [:,:,0]=B, [:,:,1]=G, [:,:,2]=R
        dx =  diff[:, :, 1] / 255.0  # Green 通道
        dy = (diff[:, :, 0] - diff[:, :, 2]) / 255.0  # B - R
        
        return dx, dy
    
    def img2grad_nonlinear(self, frame0, frame, bias=1.0):
        """
        带非线性变换的梯度计算（原始pose.py方法）
        注意：/32 会使梯度值非常小，仅用于特定校准的传感器
        """
        diff = frame.astype(np.float32) - frame0.astype(np.float32)
        diff = diff * bias
        
        dx = diff[:, :, 1] / 255.0
        dy = (diff[:, :, 0] - diff[:, :, 2]) / 255.0
        
        # 非线性变换
        dx = np.clip(dx, -0.99, 0.99)
        dy = np.clip(dy, -0.99, 0.99)
        
        dx = dx / np.sqrt(1 - dx ** 2)
        dy = dy / np.sqrt(1 - dy ** 2)
        
        return dx, dy
    
    def img2depth(self, frame0, frame, bias=1.0, x_ratio=0.5, y_ratio=0.5):
        """
        从图像计算深度（使用 recon_py 函数）
        
        Args:
            frame0: 参考帧
            frame: 当前帧
            bias: 偏置系数
            x_ratio: x方向比例系数
            y_ratio: y方向比例系数
            
        Returns:
            depth: 深度图
            dx, dy: 梯度
        """
        # 使用新的 recon_py 函数
        depth, dx, dy = recon_py(frame, frame0, x_ratio, y_ratio, bias)
        
        return depth, dx, dy
    
    def process_frame(self, frame):
        """
        处理单帧图像
        
        Args:
            frame: BGR 格式图像
            
        Returns:
            depth_colored: 深度图可视化
            normal_colored: 法向量可视化
            raw_depth: 原始深度数据
            raw_normals: 原始法向量数据
            grad_x, grad_y: 梯度数据
            diff_img: 差分图（用于调试）
        """
        # 应用透视变换
        warped = self.warp_perspective(frame)
        
        # 裁剪边缘
        raw_image = self._crop_image(warped)
        
        if self.con_flag:
            # 第一帧作为参考
            self.ref_frame = raw_image.copy()
            self.ref_blur = cv2.GaussianBlur(self.ref_frame.astype(np.float32), (13, 13), 0)
            self.con_flag = False
            
            # 返回空结果
            h, w = raw_image.shape[:2]
            return (
                np.zeros((h, w, 3), dtype=np.uint8),
                np.zeros((h, w, 3), dtype=np.uint8),
                None, None, None, None, None
            )
        
        # 当前帧模糊处理
        frame_blur = cv2.GaussianBlur(raw_image.astype(np.float32), (5, 5), 0)
        
        # 计算差分图（用于显示）
        diff = frame_blur - self.ref_blur
        diff_display = np.clip(diff * 2 + 127, 0, 255).astype(np.uint8)
        
        # 使用梯度方法计算深度
        depth, grad_x, grad_y = self.img2depth(self.ref_blur, frame_blur)
        
        # 打印梯度统计信息（每10帧打印一次）
        if hasattr(self, 'frame_count'):
            self.frame_count += 1
        else:
            self.frame_count = 0
            
        if self.frame_count % 30 == 0:
            grad_x_valid = grad_x[np.abs(grad_x) > 0.01]
            grad_y_valid = grad_y[np.abs(grad_y) > 0.01]
            if len(grad_x_valid) > 0:
                print("=" * 50)
                print("[梯度法] 统计信息：")
                print(f"  grad_x: min={np.min(grad_x):.4f}, max={np.max(grad_x):.4f}")
                print(f"  grad_y: min={np.min(grad_y):.4f}, max={np.max(grad_y):.4f}")
                print(f"  depth:  min={np.min(depth):.4f}, max={np.max(depth):.4f}")
                print("=" * 50)
        
        # 计算法向量（放大梯度使法向量变化更明显）
        grad_scale = 5.0  # 放大系数，可调整
        gx_scaled = grad_x * grad_scale
        gy_scaled = grad_y * grad_scale
        
        denom = np.sqrt(1.0 + gx_scaled**2 + gy_scaled**2)
        normal_x = -gx_scaled / denom
        normal_y = -gy_scaled / denom
        normal_z =  1.0 / denom
        normal_vector = np.stack([normal_x, normal_y, normal_z], axis=-1)
        
        # 法向量可视化
        N_disp = 0.5 * (normal_vector + 1.0)
        N_disp = np.clip(N_disp, 0, 1)
        normal_colored = (N_disp * 255).astype(np.uint8)
        normal_colored = cv2.cvtColor(normal_colored, cv2.COLOR_RGB2BGR)
        
        # 深度归一化和可视化
        depth_normalized = depth - np.min(depth)
        if np.max(depth_normalized) > 0:
            depth_normalized = depth_normalized / np.max(depth_normalized) * 255
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # 应用颜色映射
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
        
        return depth_colored, normal_colored, depth, normal_vector, grad_x, grad_y, diff_display
    
    def reset(self):
        """重置处理器，下一帧将作为新的参考帧"""
        self.con_flag = True
        self.ref_frame = None
        self.ref_blur = None
        print("[INFO] 处理器已重置，下一帧将作为参考帧")


def main():
    """主函数"""
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
        r_gain=1.1,
        g_gain=1.0,
        b_gain=1.0
    )
    
    camera = TactileCamera(camera_config)
    processor = GradientProcessor(pad=10, sensor_id="right")
    
    # 数据保存目录
    save_dir = "/home/qinxuhao/lerobot/src/lerobot/cameras/tactile_cam/tactile_data"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        camera.connect()
        print("[INFO] 相机已连接")
        
        # 创建窗口
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Difference', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth (Gradient)', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Normal Vector', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Gradient X', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Gradient Y', cv2.WINDOW_NORMAL)
        
        print("\n=== 基于梯度的触觉传感器测试 ===")
        print("操作说明:")
        print("  r - 重置参考帧")
        print("  s - 保存当前数据")
        print("  q - 退出")
        print("================================\n")
        
        # 数据收集
        all_depth = []
        all_normals = []
        all_gradients = []
        frame_count = 0
        SAVE_EVERY = 10
        
        while True:
            try:
                frame = camera.async_read(timeout_ms=200)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 处理帧
                result = processor.process_frame(frame_bgr)
                depth_colored, normal_colored, raw_depth, raw_normals, grad_x, grad_y, diff_img = result
                
                # 显示原始图像（透视变换后）
                warped = processor.warp_perspective(frame_bgr)
                cv2.imshow('Original', warped)
                
                # 显示差分图
                if diff_img is not None:
                    cv2.imshow('Difference', diff_img)
                
                # 显示深度和法向量
                cv2.imshow('Depth (Gradient)', depth_colored)
                cv2.imshow('Normal Vector', normal_colored)
                
                # 显示梯度（如果有）
                if grad_x is not None:
                    # 梯度可视化 - 使用颜色映射更直观
                    # 归一化到 [0, 255]
                    grad_x_norm = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX)
                    grad_y_norm = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX)
                    
                    grad_x_vis = cv2.applyColorMap(grad_x_norm.astype(np.uint8), cv2.COLORMAP_JET)
                    grad_y_vis = cv2.applyColorMap(grad_y_norm.astype(np.uint8), cv2.COLORMAP_JET)
                    
                    cv2.imshow('Gradient X', grad_x_vis)
                    cv2.imshow('Gradient Y', grad_y_vis)
                    
                    # 收集数据
                    frame_count += 1
                    if frame_count % SAVE_EVERY == 0 and raw_depth is not None:
                        all_depth.append(raw_depth.copy())
                        all_normals.append(raw_normals.copy())
                        all_gradients.append(np.stack([grad_x, grad_y], axis=-1))
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('r'):
                    processor.reset()
                    
                elif key == ord('s'):
                    # 手动保存
                    if raw_depth is not None:
                        timestamp = int(cv2.getTickCount())
                        # np.save(os.path.join(save_dir, f"depth_grad_{timestamp}.npy"), raw_depth)
                        # np.save(os.path.join(save_dir, f"normal_grad_{timestamp}.npy"), raw_normals)
                        # np.save(os.path.join(save_dir, f"gradient_{timestamp}.npy"), 
                        #        np.stack([grad_x, grad_y], axis=-1))
                        print(f"[INFO] 数据已保存: depth_grad_{timestamp}.npy")
                    
                elif key == ord('q'):
                    break
                    
            except TimeoutError:
                continue
            except RuntimeError as e:
                print(f"[WARNING] 帧读取错误: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    
    except Exception as e:
        print(f"[ERROR] 发生错误: {e}")
    
    finally:
        # 保存收集的数据
        if 'all_depth' in dir() and len(all_depth) > 0:
            np.save(os.path.join(save_dir, "depth_gradient.npy"), np.array(all_depth))
            np.save(os.path.join(save_dir, "normals_gradient.npy"), np.array(all_normals))
            np.save(os.path.join(save_dir, "gradients.npy"), np.array(all_gradients))
            print(f"[INFO] 保存了 {len(all_depth)} 帧数据到 {save_dir}")
        
        cv2.destroyAllWindows()
        if 'camera' in dir():
            camera.disconnect()
            print("[INFO] 相机已断开")


if __name__ == "__main__":
    main()
