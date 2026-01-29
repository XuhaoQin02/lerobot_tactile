from lerobot.cameras.tactile_cam.tactile_camera import TactileCamera
from lerobot.cameras.tactile_cam.tactile_config import TactileCameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation
import cv2
import numpy as np
import os
from lerobot.cameras.tactile_cam.gelsight_marker_tracker import GelSightMarkerTracker
from lerobot.cameras.tactile_cam.fast_poisson import fast_poisson

class GelSightProcessor:
    
    def __init__(self, table_path=None, pad=20):

        self.pad = pad
        self.zeropoint = [-65, -55, -120]
        self.lookscale = [125, 108, 260] 
        self.bin_num = 90
        self.scale = 1

        if table_path is None:
            # current_dir = os.path.dirname(os.path.abspath(__file__))
            # table_path = os.path.join(current_dir, "load", "table_3_smooth.npy")
            # hardcode temp
            table_path = "/home/donghy/lerobot/src/lerobot/cameras/tactile_cam/load/table_3_smooth.npy"
        
        self.table = np.load(table_path)
        
        self.ref_blur = None
        self.blur_inverse = None
        self.red_mask = None
        self.dmask = None
        self.kernel = self._make_kernel(9, 'circle')
        self.kernel2 = self._make_kernel(9, 'circle')
        self.con_flag = True
        self.reset_shape = True
        self.homography_matrix = None
        self.calib_file = "/home/donghy/lerobot/src/lerobot/cameras/tactile_cam/calibration_data/homography_matrix.npz"
    
    def process_image(self, image):
        """
        基于预加载的透视矩阵矫正图像并保存
        Args:
            image: 输入BGR图像
        """
        if self.homography_matrix is None:
            print(f'[WARNING]  透视变换矩阵未加载，跳过保存')
            return
        warped_img = cv2.warpPerspective(
            image, 
            self.homography_matrix, 
            (image.shape[1], image.shape[0])  
        )
        return warped_img
    
    def _make_kernel(self, n, k_type):
        if k_type == 'circle':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
        return kernel
    
    def _crop_image(self, img, pad):
        return img[pad:-pad, pad:-pad]
    
    # 缺陷区域都直接crop了，还要掩码干嘛？
    # 输入单通道即可，因为只需要HW信息
    def _defect_mask(self, img):
        pad = 20
        im_mask = np.ones((img.shape))
        im_mask[:pad, :] = 0
        im_mask[-pad:, :] = 0
        im_mask[:, :pad * 2 + 20] = 0
        im_mask[:, -pad:] = 0
        return im_mask.astype(int)
    
    # 差分图像差异大于某个阈值，被检测为marker。
    def _marker_detection(self, raw_image):
        m, n = raw_image.shape[1], raw_image.shape[0]
        raw_image_blur = cv2.GaussianBlur(raw_image.astype(np.float32), (5, 5), 0)
        ref_blur = cv2.GaussianBlur(raw_image.astype(np.float32), (25, 25), 0)
        diff = ref_blur - raw_image_blur
        diff *= 16.0
        diff[diff < 0.] = 0.
        diff[diff > 255.] = 255.
        mask = ((diff[:, :, 0] > 25) & (diff[:, :, 2] > 25) & (diff[:, :, 1] > 120))
        mask = cv2.resize(mask.astype(np.uint8), (m, n))
        mask = cv2.dilate(mask, self.kernel2, iterations=1)
        return mask
    
    # marker_detection的输出二值图作为输入，找到标记点，作为make_mask的输入，输出标记点处的掩码
    def _find_dots(self, binary_image):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 1
        params.maxThreshold = 12
        params.minDistBetweenBlobs = 9
        params.filterByArea = True
        params.minArea = 5
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minInertiaRatio = 0.5
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary_image.astype(np.uint8))
        return keypoints
    
    def _make_mask(self, img, keypoints):
        img_mask = np.zeros_like(img[:, :, 0])
        for i in range(len(keypoints)):
            cv2.ellipse(img_mask,
                       (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])),
                       (9, 6), 0, 0, 360, (1), -1)
        return img_mask

    def _matching_v2(self, test_img, ref_blur, blur_inverse):
        # 图像差异
        diff_temp1 = test_img - ref_blur
        # 差异增强
        diff_temp2 = diff_temp1 * blur_inverse
        
        # 分通道归一化
        diff_temp3 = np.zeros_like(diff_temp2, dtype=np.float32)
        for ch in range(3):  # 遍历 0=Blue、1=Green、2=Red 三通道
            # 每个通道使用对应的 zeropoint 和 lookscale
            diff_temp3[..., ch] = (diff_temp2[..., ch] - self.zeropoint[ch]) / self.lookscale[ch]
        
        # 归一化后裁剪到[0, 0.999]，避免索引越界
        diff_temp3 = np.clip(diff_temp3, 0, 0.999)
        # 转化为索引
        diff = (diff_temp3 * self.bin_num).astype(int)
        diff = np.clip(diff, 0, self.bin_num-1)
        # 查找到 grad_img
        grad_img = self.table[diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], :]
        return grad_img
    
    # 输入的是 warped image
    def process_frame(self, frame):

        raw_image = self._crop_image(frame, self.pad)
        
        if self.con_flag:
            # 第一帧作为参考图像
            ref_image = raw_image.copy()
            marker = self._marker_detection(ref_image.copy())
            keypoints = self._find_dots((1 - marker) * 255)
            
            if self.reset_shape:
                marker_mask = self._make_mask(ref_image.copy(), keypoints)
                ref_image = cv2.inpaint(ref_image, marker_mask, 3, cv2.INPAINT_TELEA)
                
                # 红色通道掩膜：只保留红色纹理区域（GelSight表面的有效触觉区域）
                self.red_mask = (ref_image[:, :, 2] > 12).astype(np.uint8)
                # 缺陷掩膜
                self.dmask = self._defect_mask(ref_image[:, :, 0])
                # 参考模糊图
                self.ref_blur = cv2.GaussianBlur(ref_image.astype(np.float32), (5, 5), 0)
                # 模糊逆权重
                self.blur_inverse = 1 + ((np.mean(self.ref_blur) / (self.ref_blur + 1)) - 1) * 2
                
                self.reset_shape = False
            
            self.con_flag = False
            
            # 返回空结果，因为第一帧只用于初始化
            depth_colored = np.zeros((raw_image.shape[0], raw_image.shape[1], 3), dtype=np.uint8)
            normal_colored = np.zeros((raw_image.shape[0], raw_image.shape[1], 3), dtype=np.uint8)
            raw_depth = None
            raw_normals = None
            return depth_colored, normal_colored, raw_depth, raw_normals
        
        else:
            # 处理阶段：计算法向量和深度
            # 与参考图像一致的高斯模糊 (3, 3) sigma自动计算
            raw_image = cv2.GaussianBlur(raw_image.astype(np.float32), (5, 5), 0)
            
            # 检测标记点 * self.dmask
            marker_mask = self._marker_detection(raw_image) 
            
            # 使用查找表计算梯度
            grad_img2 = self._matching_v2(raw_image, self.ref_blur, self.blur_inverse)
            
            # 提取x和y方向梯度，并应用掩膜 * self.red_mask
            grad_x = grad_img2[:, :, 0] * (1 - marker_mask) 
            grad_y = grad_img2[:, :, 1] * (1 - marker_mask)

            grad_x_smoothed = cv2.GaussianBlur(grad_x, (5,5), sigmaX=0)
            grad_y_smoothed = cv2.GaussianBlur(grad_y, (5,5), sigmaX=0)
            grad_x_smoothed = grad_x_smoothed * (1 - marker_mask)
            grad_y_smoothed = grad_y_smoothed * (1 - marker_mask)

            grad_x_valid = grad_x[grad_x != 0] 
            grad_y_valid = grad_y[grad_y != 0]
            if len(grad_x_valid) > 0 and len(grad_y_valid) > 0:
                print("="*50)
                print("梯度X方向统计信息：")
                print(f"  最大值: {np.max(grad_x_valid):.4f}")
                print(f"  最小值: {np.min(grad_x_valid):.4f}")
                print("梯度Y方向统计信息：")
                print(f"  最大值: {np.max(grad_y_valid):.4f}")
                print(f"  最小值: {np.min(grad_y_valid):.4f}")
                print("="*50)          
            
            # 计算法向量
            denom = np.sqrt(1.0 + grad_x**2 + grad_y**2)
            normal_x = -grad_x / denom
            normal_y = -grad_y / denom
            normal_z = 1.0 / denom
            normal_vector = np.stack([normal_x, normal_y, normal_z], axis=-1)
            
            # 将法向量转换为RGB可视化 (归一化到[0,1])
            N_disp = 0.5 * (normal_vector + 1.0)
            N_disp = np.clip(N_disp, 0, 1)
            N_disp_cv2 = (N_disp * 255).astype(np.uint8)
            N_disp_cv2 = cv2.cvtColor(N_disp_cv2, cv2.COLOR_RGB2BGR)
            
            # 计算深度 (通过泊松方程从梯度恢复深度)
            depth = fast_poisson(grad_x_smoothed, grad_y_smoothed)
            
            # 深度归一化和可视化
            depth_min = np.nanmin(depth[depth != 0]) if np.any(depth != 0) else 0
            depth = depth - depth_min
            depth[depth < 0] = 0

            depth_denoised = cv2.bilateralFilter(depth.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)
            
            depth_normalized = cv2.normalize(depth_denoised, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = depth_normalized.astype(np.uint8)
            
            # 应用颜色映射
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
            
            return depth_colored, N_disp_cv2, depth, normal_vector

def main():
    ov5647_config = TactileCameraConfig(
        index_or_path="/dev/video2", 
        fps=25,                       
        width=640,                   
        height=480,
        color_mode=ColorMode.RGB,     
        rotation=Cv2Rotation.NO_ROTATION, 
    )

    ov5647_camera = TactileCamera(ov5647_config)
    processor = GelSightProcessor()
    marker_tracker = GelSightMarkerTracker()
    
    try:
        ov5647_camera.connect()
        
        print("Initializing...")
        
        cv2.namedWindow('Tactile Camera Frame', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Normal Vector', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Marker Motion', cv2.WINDOW_NORMAL)
        
        # 设置窗口大小
        cv2.resizeWindow('Tactile Camera Frame', 640, 480)
        cv2.resizeWindow('Depth Map', 640, 480)
        cv2.resizeWindow('Normal Vector', 640, 480)
        cv2.resizeWindow('Marker Motion', 640, 480)

        marker_initialized = False
        frame_count = 0

        # SAVE_DIR = "tactile_data"
        # os.makedirs(SAVE_DIR, exist_ok=True)
        TARGET_DIR = "/home/donghy/lerobot/src/lerobot/cameras/tactile_cam/tactile_data"

        all_normal_maps = []
        all_depth_maps = []
        all_displacements = []
        SAVE_EVERY = 10
        
        while True:
            try:
                frame = ov5647_camera.async_read(timeout_ms=200)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                calib_data = np.load(processor.calib_file)
                processor.homography_matrix = calib_data['homography_matrix']
                warped_frame_bgr = processor.process_image(frame_bgr)

                if not marker_initialized and frame_count > 5:
                    marker_tracker.reinit(warped_frame_bgr)
                    marker_tracker.start_display_markerIm()
                    marker_initialized = True
                    print("Marker tracker initialized")

                depth_colored, normal_colored, raw_depth, raw_normals = processor.process_frame(warped_frame_bgr)

                if marker_initialized:
                    marker_tracker.update_markerMotion(warped_frame_bgr)
                displacements = marker_tracker.get_marker_displacements() 

                if not processor.con_flag and raw_depth is not None and raw_normals is not None:
                    frame_count += 1
                    if frame_count % SAVE_EVERY == 0:
                        all_normal_maps.append(raw_normals.copy())
                        all_depth_maps.append(raw_depth.copy())
                        all_displacements.append(displacements.copy() if displacements is not None else np.array([]))
                    

                cv2.imshow('Tactile Camera Frame', warped_frame_bgr)
                cv2.imshow('Depth Map', depth_colored)
                cv2.imshow('Normal Vector', normal_colored)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            except TimeoutError as e:
                print(f"帧读取超时: {e}，重试...")
                continue
            except RuntimeError as e:
                print(f"帧读取运行时错误: {e}，重试...")
                continue
    
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        if len(all_normal_maps) > 0:
            normal_array = np.array(all_normal_maps)
            depth_array = np.array(all_depth_maps)
            
            np.save(os.path.join(TARGET_DIR, "normals.npy"), normal_array)
            np.save(os.path.join(TARGET_DIR, "depth.npy"), depth_array)
            np.save(os.path.join(TARGET_DIR, "displacements.npy"), all_displacements, allow_pickle=True)
            
            print(f"最终保存 {len(all_normal_maps)} 帧数据")
        cv2.destroyAllWindows() 
        marker_tracker.cleanup()
        ov5647_camera.disconnect()
        print("相机已断开连接")

if __name__ == "__main__":
    main()


