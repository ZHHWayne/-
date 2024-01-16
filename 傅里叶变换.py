import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian
    return np.clip(noisy_image, 0, 255)

def fft_denoise(image, threshold=0.1):
    # 进行傅里叶变换
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # 计算频谱
    magnitude_spectrum = np.log(1 + np.abs(f_transform_shifted))

    # 构建一个低通滤波器
    rows, cols, channels = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, channels), np.uint8)
    r = 30  # 半径，可以根据需要调整
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    # 应用滤波器
    f_transform_shifted = f_transform_shifted * mask

    # 反转移位
    f_transform_inverse = np.fft.ifftshift(f_transform_shifted)
    img_denoised = np.fft.ifft2(f_transform_inverse).real

    # 返回去噪后的图像
    return np.clip(img_denoised, 0, 255).astype(np.uint8)

# 读取图像
original_image = cv2.imread('./dataset/Set12/02.png')

# 添加高斯噪声
noisy_image_gaussian = add_gaussian_noise(original_image)

# 进行傅里叶变换去噪
denoised_image_gaussian = fft_denoise(noisy_image_gaussian)

# 显示图像和去噪效果
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(original_image)
axs[0].set_title('Original Image')

axs[1].imshow(noisy_image_gaussian)
axs[1].set_title('Noisy Image (Gaussian)')

axs[2].imshow(denoised_image_gaussian)
axs[2].set_title('Denoised Image (Gaussian)')

for ax in axs:
    ax.axis('off')

plt.show()


