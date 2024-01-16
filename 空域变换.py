import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

# 读取图像
img = cv2.imread('./dataset/BSD68/')

# 添加高斯噪声
mean = 0
var = 0.1
sigma = var ** 0.5
# 生成符合高斯分布的随机噪声，均值为0，标准差为sigma
gaussian = np.random.normal(mean, sigma, img.shape)
# 将噪声的形状改为与原始图像相同
gaussian = gaussian.reshape(img.shape)
# 将随机噪声添加到原始图像中
noisy_img = img + gaussian

# 添加椒盐噪声
s_vs_p = 0.5
amount = 0.05
# 计算添加的椒盐噪声点的数量
num_salt = np.ceil(amount * img.size * s_vs_p)
num_pepper = np.ceil(amount * img.size * (1.0 - s_vs_p))
# 生成坐标数组，用于添加椒盐噪声
coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
# 将椒盐噪声点的像素值分别设置为白色和黑色
noisy_img[coords_salt[0], coords_salt[1], :] = 255
noisy_img[coords_pepper[0], coords_pepper[1], :] = 0

# 将像素值限制在 0 和 255 之间
noisy_img = np.clip(noisy_img, 0, 255)

# 去除高斯噪声
dst = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("1",img)
cv2.imshow('Original Image', noisy_img)

def display_denoise_effect(original, denoised, box_position, box_size, save_path=None):
    fig, ax = plt.subplots()
    original = original.astype(np.uint8)
    denoised = denoised.astype(np.uint8)
    # 显示整体去噪效果
    ax.imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    ax.axis('off')  # 关闭坐标轴
    # 添加小框
    rect = Rectangle((box_position[0], box_position[1]), box_size[0], box_size[1], linewidth=2, edgecolor='g',
                     facecolor='none')
    ax.add_patch(rect)

    # 提取小框内的图像
    denoised_box = denoised[box_position[1]:box_position[1] + box_size[1],
                   box_position[0]:box_position[0] + box_size[0]]

    # 在左下角显示小框内的去噪效果
    ax_inset = fig.add_axes([0.5, 0.11, 0.25, 0.25])  # 调整放大图的位置和大小
    ax_inset.imshow(cv2.cvtColor(denoised_box, cv2.COLOR_BGR2RGB))
    ax_inset.axis('off')

    # 保存图像
    if save_path:
        plt.savefig(save_path)
        print(f'Saved result to {save_path}')

    plt.show()


# 示例使用
original_image = img
denoised_image = dst

# 设置小框的位置和大小
box_position = (100, 100)
box_size = (50, 50)

# 保存图像的文件路径
save_path = 'jieguo'

# 调用函数显示图像和效果，并保存结果
display_denoise_effect(original_image, denoised_image, box_position, box_size, save_path)

def evaluate_denoising(original, denoised):
    mse_value = mse(original, denoised)
    psnr_value = psnr(original, denoised, data_range=original.max() - original.min())
    ssim_value, _ = ssim(original, denoised, win_size=3, full=True, data_range=original.max() - original.min())

    return mse_value, psnr_value, ssim_value

# 示例使用
original_image = img
denoised_image = dst

mse_value, psnr_value, ssim_value = evaluate_denoising(original_image, denoised_image)

print(f'Mean Squared Error (MSE): {mse_value}')
print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr_value} dB')
print(f'Structural Similarity Index (SSIM): {ssim_value}')