import numpy as np
import cv2

homography_matrix_file = "/home/donghy/lerobot/src/lerobot/cameras/tactile_cam/calibration_data/homography_matrix.npz"

# def affine_transform(img):
#     # pto_o = np.float32([[33, 58], [208, 61], [56, 250], [185, 250]])
#     pto_o = np.float32([[33, 58], [208, 61], [56, 250], [185, 244]])
#     pts_d = np.float32([[0, 0], [240, 0], [0, 320], [240, 320]])
#     M = cv2.getPerspectiveTransform(pto_o, pts_d)

#     img_ = cv2.warpPerspective(img, M, (240, 320))

#     return img_

def affine_transform(img):
    """
    应用透视变换到图像,基于标定文件中的单应矩阵

    Args:
        img: 输入图像 (numpy array) 

    Returns:
        img_transformed: 变换后的图像 (numpy array) 
    """

    calib_data = np.load(homography_matrix_file)
    homography_matrix = calib_data['homography_matrix']
    output_size = tuple(int(x) for x in calib_data['output_size'])

    img_transformed = cv2.warpPerspective(
        img, 
        homography_matrix, 
        output_size,
        flags=cv2.INTER_NEAREST
    )

    return img_transformed


if __name__ == "__main__":
    img = cv2.imread('data/img.jpg')
    # img_ = img[78:235,57:184]
    # cv2.imshow("ref", img_)

    # [57,78],[184,78],[69,235],[169,235]
    # [33,63],[213,63],[54,248],[187,248]

    img_ = affine_transform(img)

    cv2.imwrite('data/img_.jpg', img_)

    cv2.imshow("img", img)
    cv2.imshow("ybk", img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()