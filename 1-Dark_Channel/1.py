# import os
# import cv2
# import numpy as np
# import math
# import time
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim

# class HazeRemoval():
#     def __init__(self, omega=0.95, t0=0.1, radius=7, r=60, eps=1e-4):
#         self.omega = omega  # 透射率估计的权重参数
#         self.t0 = t0  # 最小透射值
#         self.radius = radius  # 暗通道计算半径
#         self.r = r  # 导向滤波器半径
#         self.eps = eps  # 导向滤波器的epsilon值

#     def get_dark_channel(self, img):
#         """计算暗通道"""
#         b, g, r = cv2.split(img)
#         min_rgb = cv2.min(cv2.min(r, g), b)  # 计算每个像素的RGB通道中的最小值
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * self.radius + 1, 2 * self.radius + 1))
#         dark = cv2.erode(min_rgb, kernel)  # 进行最小滤波以获取暗通道
#         return dark

#     def estimate_atmospheric_light(self, img, dark):
#         """估计大气光A"""
#         [h, w] = img.shape[:2]
#         imgsize = h * w
#         Top_pixels = int(max(math.floor(h * w / 1000), 1))  # 计算暗通道中最亮的前0.1%像素数量

#         darkvec = dark.reshape(imgsize)
#         imgvec = img.reshape(imgsize, 3)

#         indexes = darkvec.argsort()
#         indexes = indexes[imgsize - Top_pixels::]  # 选择对应于暗通道中最亮的前0.1%像素的索引

#         A = np.zeros(3)
#         for i in range(Top_pixels):
#             pixel = imgvec[indexes[i], :]  # 获取所选索引处像素的RGB值
#             for channel in range(img.shape[2]):
#                 if pixel[channel] > A[channel]:
#                     A[channel] = pixel[channel]  # 如果当前像素值大于存储值,用当前像素值更新存储值
#         return A

#     def estimate_transmission(self, img, A):
#         """估计透射率t"""
#         I_dark = np.empty(img.shape, img.dtype)

#         for channel in range(img.shape[2]):
#             I_dark[:, :, channel] = img[:, :, channel] / A[channel]  # 将每个颜色通道除以相应的大气光值

#         transmission = 1 - self.omega * self.get_dark_channel(I_dark)  # 根据公式，使用暗通道先验估计透射率
#         return transmission

#     def guided_filter(self, img, p):
#         """导向滤波"""
#         m_I = cv2.boxFilter(img, cv2.CV_64F, (self.r, self.r))
#         m_p = cv2.boxFilter(p, cv2.CV_64F, (self.r, self.r))
#         m_Ip = cv2.boxFilter(img * p, cv2.CV_64F, (self.r, self.r))  # 均值滤波
#         cov_Ip = m_Ip - m_I * m_p  # 协方差

#         m_II = cv2.boxFilter(img * img, cv2.CV_64F, (self.r, self.r))  # 均值滤波
#         var_I = m_II - m_I * m_I  # 方差

#         a = cov_Ip / (var_I + self.eps)  # 系数 a
#         b = m_p - a * m_I  # 系数 b

#         m_a = cv2.boxFilter(a, cv2.CV_64F, (self.r, self.r))  # 对系数 a 进行均值滤波
#         m_b = cv2.boxFilter(b, cv2.CV_64F, (self.r, self.r))  # 对系数 b 进行均值滤波

#         q = m_a * img + m_b  # 计算导向滤波结果
#         return q

#     def refine_transmission(self, img, t):
#         """透射率精炼"""
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gray = np.float64(gray) / 255
#         refined_t = self.guided_filter(gray, t)  # 对透射率进行导向滤波
#         return refined_t

#     def recover_image(self, img, t, A):
#         """恢复去雾图像"""
#         recovered_img = np.empty(img.shape, img.dtype)
#         t = cv2.max(t, self.t0)  # 透射率取一个最小值

#         for channel in range(img.shape[2]):
#             recovered_img[:, :, channel] = (img[:, :, channel] - A[channel]) / t + A[channel]  # 恢复去雾图像

#         return recovered_img * 255

#     def recover_depth_map(self, transmission, A):
#         """深度估计"""
#         depth_map = -0.1 * np.log(transmission) * 255
#         depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
#         depth_map_normalized_gray = (depth_map_normalized * 255).astype('uint8')
#         depth_map_heatmap = cv2.applyColorMap(depth_map_normalized_gray, cv2.COLORMAP_HOT)
#         return depth_map_heatmap

# def plot_metrics_to_pdf(file_indexes, metrics_dict, output_folder, output_pdf):
#     with PdfPages(os.path.join(output_folder, output_pdf)) as pdf:
#         for metric_name, metric_list in metrics_dict.items():
#             plt.figure()
#             plt.plot(file_indexes, metric_list, linestyle='-', label=metric_name)
#             plt.xlabel('File Index')
#             plt.ylabel(metric_name)
#             plt.title(f'{metric_name} values for first {len(file_indexes)} files')
#             plt.legend()
#             pdf.savefig()  # Save the current figure into the PDF
#             plt.close()

# def get_gt_filename(hazy_filename):
#     # 假设无雾图像文件名通过去掉有雾图像文件名中的特定部分生成，例如去掉后缀或特定子字符串
#     # 例如：'1434_8.png' -> '1434.png'
#     base_name = hazy_filename.split("_")[0]
#     gt_filename = f"{base_name}.png"
#     return gt_filename

# origin_path = "./SOTS/HR_hazy"
# gt_path = "./SOTS/HR"
# hazedark_folder_path = './SOTS/HR_haze_dark'
# gtdark_folder_path = './SOTS/HR_gt_dark'
# recover_folder_path = "./SOTS/HR_recover"
# result_file_path = "./SOTS/result.txt"
# depth_folder_path = "./SOTS/Depth"
# pdf_output_file = "dehazing_metrics.pdf"

# output_folders = [
#     hazedark_folder_path, gtdark_folder_path,
#     recover_folder_path, depth_folder_path
# ]

# for folder in output_folders:
#     if not os.path.exists(folder):
#         os.makedirs(folder)

# psnr_500 = []
# ssim_500 = []
# time_500 = []
# psnr_1000 = []
# ssim_1000 = []
# time_1000 = []

# i = 0
# file_indexes = []

# with open(result_file_path, "w") as f:
#     for root, dirs, files in os.walk(origin_path):
#         for filename in files:
#             file_path = os.path.join(origin_path, filename)
#             gt_filename = get_gt_filename(filename)
#             gt_file_path = os.path.join(gt_path, gt_filename)
#             hazedark_file_path = os.path.join(hazedark_folder_path, filename.split(".")[0] + ".png")
#             gtdark_file_path = os.path.join(gtdark_folder_path, gt_filename.split(".")[0] + ".png")
#             recover_file_path = os.path.join(recover_folder_path, filename.split(".")[0] + ".png")
#             depth_file_path = os.path.join(depth_folder_path, filename.split(".")[0] + ".png")

#             print(f"Processing file: {file_path}")
#             print(f"Ground truth file: {gt_file_path}")

#             img = cv2.imread(file_path)
#             gt = cv2.imread(gt_file_path)

#             start = time.time()
#             haze_removal = HazeRemoval()

#             dark_channel = haze_removal.get_dark_channel(img)
#             cv2.imwrite(hazedark_file_path, dark_channel)

#             A = haze_removal.estimate_atmospheric_light(img, dark_channel)
#             transmission = haze_removal.estimate_transmission(img, A)
#             refined_transmission = haze_removal.refine_transmission(img, transmission)
#             dehazed_img = haze_removal.recover_image(img / 255.0, refined_transmission, A)
#             cv2.imwrite(recover_file_path, dehazed_img)

#             depth_map = haze_removal.recover_depth_map(refined_transmission, A)
#             cv2.imwrite(depth_file_path, depth_map)

#             end = time.time()
#             processing_time = end - start

#             if gt is not None:
#                 gt_dark_channel = haze_removal.get_dark_channel(gt)
#                 cv2.imwrite(gtdark_file_path, gt_dark_channel)

#                 psnr_val = psnr(gt, dehazed_img)
#                 ssim_val = ssim(gt, dehazed_img, multichannel=True)

#                 if i < 500:
#                     psnr_500.append(psnr_val)
#                     ssim_500.append(ssim_val)
#                     time_500.append(processing_time)
#                 else:
#                     psnr_1000.append(psnr_val)
#                     ssim_1000.append(ssim_val)
#                     time_1000.append(processing_time)

#                 f.write(f"File: {filename}, PSNR: {psnr_val}, SSIM: {ssim_val}, Time: {processing_time}\n")

#             file_indexes.append(i)
#             i += 1

# # 统计指标并输出到文件
# def average(lst):
#     return sum(lst) / len(lst) if lst else 0

# metrics_500 = {
#     'PSNR': psnr_500,
#     'SSIM': ssim_500,
#     'Time': time_500
# }

# metrics_1000 = {
#     'PSNR': psnr_1000,
#     'SSIM': ssim_1000,
#     'Time': time_1000
# }

# # 生成指标图表并保存为PDF
# plot_metrics_to_pdf(file_indexes[:500], metrics_500, '.', 'metrics_first_500.pdf')
# plot_metrics_to_pdf(file_indexes[500:], metrics_1000, '.', 'metrics_last_1000.pdf')

# with open(result_file_path, "a") as f:
#     f.write(f"\nFirst 500 Images Average PSNR: {average(psnr_500)}, SSIM: {average(ssim_500)}, Time: {average(time_500)}\n")
#     f.write(f"Last 500 Images Average PSNR: {average(psnr_1000)}, SSIM: {average(ssim_1000)}, Time: {average(time_1000)}\n")


import os
import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class DehazeAlgorithm():
    def __init__(self, omega=0.95, t_min=0.1, dark_channel_radius=7, guided_filter_radius=60, epsilon=1e-4):
        self.omega = omega  # 透射率估计的权重参数
        self.t_min = t_min  # 最小透射值
        self.dark_channel_radius = dark_channel_radius  # 暗通道计算半径
        self.guided_filter_radius = guided_filter_radius  # 导向滤波器半径
        self.epsilon = epsilon  # 导向滤波器的epsilon值

    def compute_dark_channel(self, image):
        """计算暗通道"""
        b_channel, g_channel, r_channel = cv2.split(image)
        min_rgb = cv2.min(cv2.min(r_channel, g_channel), b_channel)  # 计算每个像素的RGB通道中的最小值
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * self.dark_channel_radius + 1, 2 * self.dark_channel_radius + 1))
        dark_channel = cv2.erode(min_rgb, kernel)  # 进行最小滤波以获取暗通道
        return dark_channel

    def estimate_atmospheric_light(self, image, dark_channel):
        """估计大气光A"""
        [height, width] = image.shape[:2]
        img_size = height * width
        top_pixels = int(max(math.floor(height * width / 1000), 1))  # 计算暗通道中最亮的前0.1%像素数量

        dark_vec = dark_channel.reshape(img_size)
        image_vec = image.reshape(img_size, 3)

        indices = dark_vec.argsort()
        indices = indices[img_size - top_pixels::]  # 选择对应于暗通道中最亮的前0.1%像素的索引

        A = np.zeros(3)
        for i in range(top_pixels):
            pixel = image_vec[indices[i], :]  # 获取所选索引处像素的RGB值
            for channel in range(image.shape[2]):
                if pixel[channel] > A[channel]:
                    A[channel] = pixel[channel]  # 如果当前像素值大于存储值,用当前像素值更新存储值
        return A

    def estimate_transmission(self, image, atmospheric_light):
        """估计透射率t"""
        I_dark = np.empty(image.shape, image.dtype)

        for channel in range(image.shape[2]):
            I_dark[:, :, channel] = image[:, :, channel] / atmospheric_light[channel]  # 将每个颜色通道除以相应的大气光值

        transmission = 1 - self.omega * self.compute_dark_channel(I_dark)  # 根据公式，使用暗通道先验估计透射率
        return transmission

    def guided_filter(self, guide_image, p_image):
        """导向滤波"""
        mean_I = cv2.boxFilter(guide_image, cv2.CV_64F, (self.guided_filter_radius, self.guided_filter_radius))
        mean_p = cv2.boxFilter(p_image, cv2.CV_64F, (self.guided_filter_radius, self.guided_filter_radius))
        mean_Ip = cv2.boxFilter(guide_image * p_image, cv2.CV_64F, (self.guided_filter_radius, self.guided_filter_radius))  # 均值滤波
        cov_Ip = mean_Ip - mean_I * mean_p  # 协方差

        mean_II = cv2.boxFilter(guide_image * guide_image, cv2.CV_64F, (self.guided_filter_radius, self.guided_filter_radius))  # 均值滤波
        var_I = mean_II - mean_I * mean_I  # 方差

        a = cov_Ip / (var_I + self.epsilon)  # 系数 a
        b = mean_p - a * mean_I  # 系数 b

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (self.guided_filter_radius, self.guided_filter_radius))  # 对系数 a 进行均值滤波
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (self.guided_filter_radius, self.guided_filter_radius))  # 对系数 b 进行均值滤波

        q = mean_a * guide_image + mean_b  # 计算导向滤波结果
        return q

    def refine_transmission(self, image, transmission):
        """透射率精炼"""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = np.float64(gray_image) / 255
        refined_transmission = self.guided_filter(gray_image, transmission)  # 对透射率进行导向滤波
        return refined_transmission

    def recover_image(self, image, transmission, atmospheric_light):
        """恢复去雾图像"""
        recovered_image = np.empty(image.shape, image.dtype)
        transmission = cv2.max(transmission, self.t_min)  # 透射率取一个最小值

        for channel in range(image.shape[2]):
            recovered_image[:, :, channel] = (image[:, :, channel] - atmospheric_light[channel]) / transmission + atmospheric_light[channel]  # 恢复去雾图像

        return recovered_image * 255

    def recover_depth_map(self, transmission, atmospheric_light):
        """深度估计"""
        depth_map = -0.1 * np.log(transmission) * 255
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        depth_map_normalized_gray = (depth_map_normalized * 255).astype('uint8')
        depth_map_heatmap = cv2.applyColorMap(depth_map_normalized_gray, cv2.COLORMAP_HOT)
        return depth_map_heatmap

def plot_metrics(metric_values, metric_name, file_indices, output_folder):
    plt.plot(file_indices, metric_values, linestyle='-', label=metric_name)
    plt.xlabel('File Index')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} values for first {len(metric_values)} files')
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'{metric_name}_plot.png'))
    plt.close()

def get_ground_truth_filename(hazy_filename):
    # 假设无雾图像文件名通过去掉有雾图像文件名中的特定部分生成，例如去掉后缀或特定子字符串
    # 例如：'1434_8.png' -> '1434.png'
    base_name = hazy_filename.split("_")[0]
    gt_filename = f"{base_name}.png"
    return gt_filename

origin_images_path = "./SOTS/HR_hazy"
gt_images_path = "./SOTS/HR"
hazy_dark_channel_path = './SOTS/HR_haze_dark'
gt_dark_channel_path = './SOTS/HR_gt_dark'
recovered_images_path = "./SOTS/HR_recover"
result_log_path = "./SOTS/result.txt"
depth_images_path = "./SOTS/Depth"
comparison_image_path = "./SOTS/comparison.pdf"

output_directories = [
    hazy_dark_channel_path, gt_dark_channel_path,
    recovered_images_path, depth_images_path
]

for directory in output_directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

psnr_indoor_list = []
ssim_indoor_list = []
time_indoor_list = []
psnr_outdoor_list = []
ssim_outdoor_list = []
time_outdoor_list = []

file_count = 0
comparison_images = []

with open(result_log_path, "w") as log_file:
    for root, dirs, files in os.walk(origin_images_path):
        for hazy_image_name in files:
            hazy_image_path = os.path.join(origin_images_path, hazy_image_name)
            gt_image_name = get_ground_truth_filename(hazy_image_name)
            gt_image_path = os.path.join(gt_images_path, gt_image_name)
            hazy_dark_image_path = os.path.join(hazy_dark_channel_path, hazy_image_name.split(".")[0] + ".png")
            gt_dark_image_path = os.path.join(gt_dark_channel_path, gt_image_name.split(".")[0] + ".png")
            recovered_image_path = os.path.join(recovered_images_path, hazy_image_name.split(".")[0] + ".png")
            depth_image_path = os.path.join(depth_images_path, hazy_image_name.split(".")[0] + ".png")

            print(f"Processing file: {hazy_image_path}")
            print(f"Ground truth file: {gt_image_path}")

            hazy_image = cv2.imread(hazy_image_path)
            gt_image = cv2.imread(gt_image_path)

            if hazy_image is None:
                print(f"Could not read hazy image: {hazy_image_path}")
                continue
            if gt_image is None:
                print(f"Could not read ground truth image: {gt_image_path}")
                continue

            dehaze = DehazeAlgorithm()
            start_time = time.time()

            hazy_image_normalized = hazy_image.astype('float64') / 255  # 将雾图像归一化

            dark_channel = dehaze.compute_dark_channel(hazy_image_normalized)  # 计算暗通道
            atmospheric_light = dehaze.estimate_atmospheric_light(hazy_image_normalized, dark_channel)  # 估计大气光
            transmission = dehaze.estimate_transmission(hazy_image_normalized, atmospheric_light)  # 估计透射率
            refined_transmission = dehaze.refine_transmission(hazy_image, transmission)  # 精炼透射率
            recovered_image = dehaze.recover_image(hazy_image_normalized, refined_transmission, atmospheric_light)  # 恢复去雾图像

            end_time = time.time()
            processing_time = end_time - start_time

            depth_map = dehaze.recover_depth_map(refined_transmission, atmospheric_light)

            file_count += 1

            cv2.imwrite(hazy_dark_image_path, dark_channel * 255)
            dark_channel_gt = dehaze.compute_dark_channel(gt_image.astype('float64') / 255)
            cv2.imwrite(gt_dark_image_path, dark_channel_gt * 255)
            cv2.imwrite(recovered_image_path, recovered_image)
            cv2.imwrite(depth_image_path, depth_map)

            psnr_value = psnr(gt_image, recovered_image, data_range=255)
            ssim_value = ssim(gt_image, recovered_image, channel_axis=2, data_range=255)

            if file_count <= 500:
                psnr_indoor_list.append(psnr_value)
                ssim_indoor_list.append(ssim_value)
                time_indoor_list.append(processing_time)
            else:
                psnr_outdoor_list.append(psnr_value)
                ssim_outdoor_list.append(ssim_value)
                time_outdoor_list.append(processing_time)

            print(f"File {hazy_image_name}: PSNR={psnr_value:.2f}, SSIM={ssim_value:.4f}, time={processing_time:.4f}s\n")
            log_file.write(f"File {hazy_image_name}: PSNR={psnr_value:.2f}, SSIM={ssim_value:.4f}, time={processing_time:.4f}s\n")

            if file_count <= 6:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(cv2.cvtColor(hazy_image, cv2.COLOR_BGR2RGB))
                axes[0].set_title("Hazy Image")
                axes[0].axis("off")

                axes[1].imshow(cv2.cvtColor(recovered_image.astype('uint8'), cv2.COLOR_BGR2RGB))
                axes[1].set_title("Recovered Image")
                axes[1].axis("off")

                axes[2].imshow(cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB))
                axes[2].set_title("Ground Truth Image")
                axes[2].axis("off")

                comparison_images.append((hazy_image, recovered_image, gt_image))

    if comparison_images:
        fig, axes = plt.subplots(len(comparison_images), 3, figsize=(15, 5 * len(comparison_images)))
        for i, (hazy_image, recovered_image, gt_image) in enumerate(comparison_images):
            axes[i, 0].imshow(cv2.cvtColor(hazy_image, cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title("Hazy Image")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(cv2.cvtColor(recovered_image.astype('uint8'), cv2.COLOR_BGR2RGB))
            axes[i, 1].set_title("Recovered Image")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB))
            axes[i, 2].set_title("Ground Truth Image")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(comparison_image_path)
        plt.close(fig)

    psnr_indoor_avg = np.mean(psnr_indoor_list)
    ssim_indoor_avg = np.mean(ssim_indoor_list)
    time_indoor_avg = np.mean(time_indoor_list)

    psnr_outdoor_avg = np.mean(psnr_outdoor_list)
    ssim_outdoor_avg = np.mean(ssim_outdoor_list)
    time_outdoor_avg = np.mean(time_outdoor_list)

    print(f"Average PSNR Indoor : {psnr_indoor_avg:.2f}, Average PSNR Outdoor : {psnr_outdoor_avg:.2f}, Average SSIM Indoor : {ssim_indoor_avg:.4f}, Average SSIM Outdoor : {ssim_outdoor_avg:.4f}, Average time Indoor : {time_indoor_avg:.4f}s, Average time Outdoor : {time_outdoor_avg:.4f}s\n")
    log_file.write(f"Average PSNR Indoor : {psnr_indoor_avg:.2f}, Average PSNR Outdoor : {psnr_outdoor_avg:.2f}, Average SSIM Indoor : {ssim_indoor_avg:.4f}, Average SSIM Outdoor : {ssim_outdoor_avg:.4f}, Average time Indoor : {time_indoor_avg:.4f}s, Average time Outdoor : {time_outdoor_avg:.4f}s\n")
