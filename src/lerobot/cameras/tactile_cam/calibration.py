import cv2
import numpy as np 
import glob
from scipy import signal
import os

class calibration:
    def __init__(self):
        self.BallRad= 8.0/2 # mm
        self.Pixmm = 0.1316 #.10577     # mm/pixel
        self.ratio = 1/2.
        self.red_range = [-90, 90]
        self.green_range = [-90, 90] #[-60, 50]
        self.blue_range = [-90, 90] # [-80, 60]
        self.red_bin = int((self.red_range[1] - self.red_range[0])*self.ratio)
        self.green_bin = int((self.green_range[1] - self.green_range[0])*self.ratio)
        self.blue_bin = int((self.blue_range[1] - self.blue_range[0])*self.ratio)
        # 用于归一化RGB差值
        self.zeropoint = [-65, -55, -120]# [-90, -90, -90]
        self.lookscale = [125, 108, 260] # [180., 180., 180.]
        self.bin_num = 90
    

    def crop_image(self, img, pad):
        return img[pad:-pad,pad:-pad]

    def mask_marker(self, raw_image):
        m, n = raw_image.shape[1], raw_image.shape[0]
        raw_image = cv2.pyrDown(raw_image).astype(np.float32)
        blur = cv2.GaussianBlur(raw_image, (25, 25), 0)
        blur2 = cv2.GaussianBlur(raw_image, (5, 5), 0)
        diff = blur - blur2
        diff *= 16.0
        # cv2.imshow('blur2', blur.astype(np.uint8))
        # cv2.waitKey(1)

        diff[diff < 0.] = 0.
        diff[diff > 255.] = 255.

        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        # cv2.imshow('diff', diff.astype(np.uint8))
        # cv2.waitKey(1)

        mask_b = diff[:, :, 0] > 150 
        mask_g = diff[:, :, 1] > 150 
        mask_r = diff[:, :, 2] > 150 
        mask = (mask_b*mask_g + mask_b*mask_r + mask_g*mask_r)>0
        # cv2.imshow('mask', mask.astype(np.uint8) * 255)
        # cv2.waitKey(1)
        mask = cv2.resize(mask.astype(np.uint8), (m, n))
#        mask = mask * self.dmask
#        mask = cv2.dilate(mask, self.kernal4, iterations=1)

        # mask = cv2.erode(mask, self.kernal4, iterations=1)
        return (1 - mask) * 255
    
    def find_dots(self, binary_image):
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 1
        params.maxThreshold = 12
        params.minDistBetweenBlobs = 9
        params.filterByArea = True
        params.minArea = 9
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minInertiaRatio = 0.5
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary_image.astype(np.uint8))
        # im_to_show = (np.stack((binary_image,)*3, axis=-1)-100)
        # for i in range(len(keypoints)):
        #     cv2.circle(im_to_show, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), 5, (0, 100, 100), -1)

        # cv2.imshow('final_image1',im_to_show)
        # cv2.waitKey(1)
        return keypoints
    
    def make_mask(self,img, keypoints):
        img = np.zeros_like(img[:,:,0])
        for i in range(len(keypoints)):
            # cv2.circle(img, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), 6, (1), -1)
            cv2.ellipse(img, (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), (9, 7) ,0 ,0 ,360, (1), -1)

        return img
    
    def contact_detection(self,raw_image, ref, marker_mask):
        blur = cv2.GaussianBlur(ref.astype(np.float32), (3, 3), 0)
        diff_img = np.max(np.abs(raw_image.astype(np.float32) - blur),axis = 2)
        contact_mask = (diff_img> 30).astype(np.uint8)*(1-marker_mask) #经验阈值
        contours,_ = cv2.findContours(contact_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        sorted_areas = np.sort(areas)
        cnt=contours[areas.index(sorted_areas[-1])] #the biggest contour
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        
        key = -1
        while key != 27:
            center = (int(x),int(y))
            radius = int(radius)
            im2show = cv2.circle(np.array(raw_image),center,radius,(0,40,0),2)
            cv2.imshow('contact', im2show.astype(np.uint8))
            key = cv2.waitKey(0)
            if key == 119:
                y -= 1
            elif key == 115:
                y += 1
            elif key == 97:
                x -= 1
            elif key == 100:
                x += 1
            elif key == 109:
                radius += 1
            elif key == 110:
                radius -= 1

        contact_mask = np.zeros_like(contact_mask)
        cv2.circle(contact_mask,center,radius,(1),-1)
        contact_mask = contact_mask * (1-marker_mask)
#        cv2.imshow('contact_mask',contact_mask*255)
#        cv2.waitKey(0)
        return contact_mask, center, radius   
    
    def get_gradient_v3(self, img, ref, center, radius_p, valid_mask, table, table_account):
        ball_radius_p = self.BallRad / self.Pixmm
        blur = cv2.GaussianBlur(ref.astype(np.float32), (3, 3), 0) + 1
        blur_inverse = 1+ ((np.mean(blur)/blur)-1)*2;
        img_smooth = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
        diff_temp1 = img_smooth - blur 
        diff_temp2 = diff_temp1 * blur_inverse

        # ========== 1. 打印原始差值（diff_temp2）的统计信息 ==========
        # 仅统计有效区域（valid_mask>0）的像素
        diff_valid = diff_temp2[valid_mask>0]
        if diff_valid.size > 0:
            print("\n===== [V2 原始差值（diff_temp2）统计] =====")
            for ch, ch_name in enumerate(['Blue', 'Green', 'Red']):
                ch_diff = diff_valid[:, ch]
                min_val = np.min(ch_diff)
                max_val = np.max(ch_diff)
                mean_val = np.mean(ch_diff)
                std_val = np.std(ch_diff)
                print(f"{ch_name}通道 - 最小值: {min_val:.2f}, 最大值: {max_val:.2f}, 均值: {mean_val:.2f}, 标准差: {std_val:.2f}")

        # ========== 2. 执行归一化，打印归一化后差值 ==========
        diff_temp2[:,:,0] = (diff_temp2[:,:,0] - self.zeropoint[0])/self.lookscale[0]
        diff_temp2[:,:,1] = (diff_temp2[:,:,1] - self.zeropoint[1])/self.lookscale[1]
        diff_temp2[:,:,2] = (diff_temp2[:,:,2] - self.zeropoint[2])/self.lookscale[2]
        
        # 统计归一化后的差值
        norm_diff_valid = diff_temp2[valid_mask>0]
        if norm_diff_valid.size > 0:
            print("\n===== [V2 归一化后差值统计] =====")
            for ch, ch_name in enumerate(['Blue', 'Green', 'Red']):
                ch_diff = norm_diff_valid[:, ch]
                min_val = np.min(ch_diff)
                max_val = np.max(ch_diff)
                # 统计被截断的像素比例（超出[0,0.999]范围）
                clip_low = np.sum(ch_diff < 0) / len(ch_diff) * 100
                clip_high = np.sum(ch_diff > 0.999) / len(ch_diff) * 100
                print(f"{ch_name}通道 - 最小值: {min_val:.2f}, 最大值: {max_val:.2f}, 低于0的像素占比: {clip_low:.2f}%, 高于0.999的像素占比: {clip_high:.2f}%")

        # ========== 3. 执行clip和分箱，打印最终索引 ==========
        diff_temp3 = np.clip(diff_temp2,0,0.999)
        diff = (diff_temp3*self.bin_num).astype(int)
        pixels_valid = diff[valid_mask>0]
        
        if pixels_valid.size > 0:
            print("\n===== [V2 最终分箱索引统计] =====")
            for ch, ch_name in enumerate(['Blue', 'Green', 'Red']):
                ch_idx = pixels_valid[:, ch]
                min_idx = np.min(ch_idx)
                max_idx = np.max(ch_idx)
                # 统计索引是否越界（理论应在0~89之间）
                out_of_bound = np.sum((ch_idx < 0) | (ch_idx >= self.bin_num)) / len(ch_idx) * 100
                print(f"{ch_name}通道 - 最小索引: {min_idx}, 最大索引: {max_idx}, 越界像素占比: {out_of_bound:.2f}%")
        print("="*50 + "\n")

        # ========== 原有逻辑继续 ==========
        x = np.linspace(0, img.shape[0]-1,img.shape[0])
        y = np.linspace(0, img.shape[1]-1,img.shape[1])
        xv, yv = np.meshgrid(y, x)
        xv = xv - center[0]
        yv = yv - center[1]
        rv = np.sqrt(xv**2 + yv**2)
        radius_p = min(radius_p, ball_radius_p-1) 
        mask = (rv < radius_p)
        mask_small = (rv < radius_p-1)
        temp = ((xv*mask)**2 + (yv*mask)**2)*self.Pixmm**2
        height_map = (np.sqrt(self.BallRad**2-temp)*mask - np.sqrt(self.BallRad**2-(radius_p*self.Pixmm)**2))*mask
        height_map[np.isnan(height_map)] = 0
        gx_num = signal.convolve2d(height_map, np.array([[0,0,0],[0.5,0,-0.5],[0,0,0]]), boundary='symm', mode='same')*mask_small
        gy_num = signal.convolve2d(height_map, np.array([[0,0,0],[0.5,0,-0.5],[0,0,0]]).T, boundary='symm', mode='same')*mask_small
        gradxseq = gx_num[valid_mask>0]
        gradyseq = gy_num[valid_mask>0]
        
        for i in range(gradxseq.shape[0]):
            b, g, r = pixels_valid[i,0], pixels_valid[i,1], pixels_valid[i,2]
            if table_account[b,g,r] < 1.: 
                table[b,g,r,0] = gradxseq[i]
                table[b,g,r,1] = gradyseq[i]
                table_account[b,g,r] += 1
            else:
                table[b,g,r,0] = (table[b,g,r,0]*table_account[b,g,r] + gradxseq[i])/(table_account[b,g,r]+1)
                table[b,g,r,1] = (table[b,g,r,1]*table_account[b,g,r] + gradyseq[i])/(table_account[b,g,r]+1)
                table_account[b,g,r] += 1
        return table, table_account


    def get_gradient_v2(self, img, ref, center, radius_p, valid_mask, table, table_account):
        ball_radius_p = self.BallRad / self.Pixmm
        blur = cv2.GaussianBlur(ref.astype(np.float32), (3, 3), 0) + 1
        blur_inverse = 1+ ((np.mean(blur)/blur)-1)*2
        img_smooth = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
        diff_temp1 = img_smooth - blur 
        diff_temp2 = diff_temp1 * blur_inverse
#        print(np.mean(diff_temp2), np.std(diff_temp2))
#        print(np.min(diff_temp2), np.max(diff_temp2))
        # 归一化到0-1之间
        diff_temp2[:,:,0] = (diff_temp2[:,:,0] - self.zeropoint[0])/self.lookscale[0]
        diff_temp2[:,:,1] = (diff_temp2[:,:,1] - self.zeropoint[1])/self.lookscale[1]
        diff_temp2[:,:,2] = (diff_temp2[:,:,2] - self.zeropoint[2])/self.lookscale[2]
        diff_temp3 = np.clip(diff_temp2,0,0.999)
        diff = (diff_temp3*self.bin_num).astype(int)
#        diff_valid = np.abs(diff * np.dstack((valid_mask,valid_mask,valid_mask)))
        pixels_valid = diff[valid_mask>0]
#        pixels_valid[:,0] = np.clip((pixels_valid[:,0] - self.blue_range[0])*self.ratio, 0, self.blue_bin-1)
#        pixels_valid[:,1] = np.clip((pixels_valid[:,1] - self.green_range[0])*self.ratio, 0, self.green_bin-1)
#        pixels_valid[:,2] = np.clip((pixels_valid[:,2] - self.red_range[0])*self.ratio, 0, self.red_bin-1)
#        pixels_valid = pixels_valid.astype(int)
        
        
#        range_blue = [max(np.mean(pixels_valid[:,0])-2*np.std(pixels_valid[:,0]),np.min(pixels_valid[:,0])), \
#                     min(np.mean(pixels_valid[:,0])+2.5*np.std(pixels_valid[:,0]), np.max(pixels_valid[:,0]))] 
#        range_green = [max(np.mean(pixels_valid[:,1])-2*np.std(pixels_valid[:,1]),np.min(pixels_valid[:,1])), \
#                     min(np.mean(pixels_valid[:,1])+2.5*np.std(pixels_valid[:,1]), np.max(pixels_valid[:,1]))] 
#        range_red = [max(np.mean(pixels_valid[:,2])-2*np.std(pixels_valid[:,2]),np.min(pixels_valid[:,2])), \
#                     min(np.mean(pixels_valid[:,2])+2.5*np.std(pixels_valid[:,2]), np.max(pixels_valid[:,2]))] 
        
#        print('blue', range_blue, 'green', range_green, 'red',  range_red)
        
#        print(np.min(pixels_valid[:,0]), np.max(pixels_valid[:,0]))
#        print(np.min(pixels_valid[:,1]), np.max(pixels_valid[:,1]))
#        print(np.min(pixels_valid[:,2]), np.max(pixels_valid[:,2]))
#        print(pixels_valid.shape)
#        plt.figure(0)
#        plt.hist(pixels_valid[:,0], bins = 256)
#        plt.figure(1)
#        plt.hist(pixels_valid[:,1], bins = 256)
#        plt.figure(2)
#        plt.hist(pixels_valid[:,2], bins = 256)
#        plt.show()
        x = np.linspace(0, img.shape[0]-1,img.shape[0])
        y = np.linspace(0, img.shape[1]-1,img.shape[1])
        xv, yv = np.meshgrid(y, x)
#        print('img shape', img.shape, xv.shape, yv.shape)
        xv = xv - center[0]
        yv = yv - center[1]
        rv = np.sqrt(xv**2 + yv**2)
        # print('radius_p', radius_p, ball_radius_p)
        radius_p = min(radius_p, ball_radius_p-1) 
        mask = (rv < radius_p)
        mask_small = (rv < radius_p-1)
#        gradmag=np.arcsin(rv*mask/ball_radius_p)*mask;
#        graddir=np.arctan2(-yv*mask, -xv*mask)*mask;
#        gradx_img=gradmag*np.cos(graddir);
#        grady_img=gradmag*np.sin(graddir);
#        depth = fast_poisson(gradx_img, grady_img)
        temp = ((xv*mask)**2 + (yv*mask)**2)*self.Pixmm**2
        height_map = (np.sqrt(self.BallRad**2-temp)*mask - np.sqrt(self.BallRad**2-(radius_p*self.Pixmm)**2))*mask
        height_map[np.isnan(height_map)] = 0
#        depth = poisson_reconstruct(grady_img, gradx_img, np.zeros(grady_img.shape))
        gx_num = signal.convolve2d(height_map, np.array([[0,0,0],[0.5,0,-0.5],[0,0,0]]), boundary='symm', mode='same')*mask_small
        gy_num = signal.convolve2d(height_map, np.array([[0,0,0],[0.5,0,-0.5],[0,0,0]]).T, boundary='symm', mode='same')*mask_small
        # depth_num = fast_poisson(gx_num, gy_num)
        # img2show = img.copy().astype(np.float64)
        # img2show[:,:,1] += depth_num*50
        # cv2.imshow('depth_img', img2show.astype(np.uint8))
        # cv2.imshow('valid_mask', valid_mask*255)
        # cv2.waitKey(0)
        gradxseq = gx_num[valid_mask>0]
        gradyseq = gy_num[valid_mask>0]
        
        for i in range(gradxseq.shape[0]):
            b, g, r = pixels_valid[i,0], pixels_valid[i,1], pixels_valid[i,2]
#            print(r,g,b)
            if table_account[b,g,r] < 1.: 
                table[b,g,r,0] = gradxseq[i]
                table[b,g,r,1] = gradyseq[i]
                table_account[b,g,r] += 1
            else:
#                print(table[b,g,r,0], gradxseq[i], table[b,g,r,1], gradyseq[i])
                table[b,g,r,0] = (table[b,g,r,0]*table_account[b,g,r] + gradxseq[i])/(table_account[b,g,r]+1)
                table[b,g,r,1] = (table[b,g,r,1]*table_account[b,g,r] + gradyseq[i])/(table_account[b,g,r]+1)
                table_account[b,g,r] += 1
        return table, table_account
        
  
    def smooth_table(self, table, count_map):
        y,x,z = np.meshgrid(np.linspace(0,self.bin_num-1,self.bin_num),
            np.linspace(0,self.bin_num-1,self.bin_num),np.linspace(0,self.bin_num-1,self.bin_num))
            
        unfill_x = x[count_map<1].astype(int)
        unfill_y = y[count_map<1].astype(int)
        unfill_z = z[count_map<1].astype(int)
        fill_x = x[count_map>0].astype(int)
        fill_y = y[count_map>0].astype(int)
        fill_z = z[count_map>0].astype(int)
        fill_gradients = table[fill_x, fill_y, fill_z,:]
        table_new = np.array(table)
        for i in range(unfill_x.shape[0]):
            distance = (unfill_x[i] - fill_x)**2 + (unfill_y[i] - fill_y)**2 + (unfill_z[i] - fill_z)**2
            if np.min(distance) < 20:
                index = np.argmin(distance)
                table_new[unfill_x[i], unfill_y[i], unfill_z[i],:] = fill_gradients[index,:]
        
        return table_new

        
        

if __name__=="__main__":
    cali = calibration()
    pad = 20

    test_data_dir = "/home/donghy/lerobot/src/lerobot/cameras/tactile_cam/test_data"
    table_save_dir = "/home/donghy/lerobot/src/lerobot/cameras/tactile_cam/load"

    ref_img = cv2.imread(os.path.join(test_data_dir, "ref.jpg"))
    ref_img = cali.crop_image(ref_img, pad)

    marker = cali.mask_marker(ref_img)
    keypoints = cali.find_dots(marker)
    marker_mask = cali.make_mask(ref_img, keypoints)
    marker_image = np.dstack((marker_mask,marker_mask,marker_mask))
    ref_img = cv2.inpaint(ref_img,marker_mask,3,cv2.INPAINT_TELEA)


    # table = np.zeros((cali.blue_bin, cali.green_bin, cali.red_bin, 2))
    # table_account = np.zeros((cali.blue_bin, cali.green_bin, cali.red_bin))
    table = np.zeros((cali.bin_num, cali.bin_num, cali.bin_num, 2))
    table_account = np.zeros((cali.bin_num, cali.bin_num, cali.bin_num))
   # cv2.imshow('ref_image', ref_img)
   # cv2.waitKey(0)
    has_marke = True 
    img_list = glob.glob(os.path.join(test_data_dir, "sample*.jpg"))
    
    for name in img_list:
#        print(name)
        img = cv2.imread(name)
        img = cali.crop_image(img, pad)
        if has_marke: 
            marker = cali.mask_marker(img)
            keypoints = cali.find_dots(marker)
            marker_mask = cali.make_mask(img, keypoints)
        else:
            marker_mask = np.zeros_like(img)
        try:
            valid_mask, center, radius_p  = cali.contact_detection(img, ref_img, marker_mask)
        except:
            os.remove(name)

        table, table_account = cali.get_gradient_v3(img, ref_img, center, radius_p, valid_mask, table, table_account)
    
    np.save(os.path.join(table_save_dir, "table_3.npy"), table)
    np.save(os.path.join(table_save_dir, "count_map.npy"), table_account)
    table = np.load(os.path.join(table_save_dir, "table_3.npy")) 
    table_account = np.load(os.path.join(table_save_dir, "count_map.npy")) 
    table_smooth = cali.smooth_table(table, table_account)
    np.save(os.path.join(table_save_dir,"table_3_smooth.npy"), table_smooth)
    print('[INFO] Calibration table is generated')