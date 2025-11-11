import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
"""
TODO Part 1: Motion blur PSF generation
"""
# 參考: https://blog.csdn.net/weixin_40522801/article/details/106454622
def generate_motion_blur_psf(img_size, length=40, angle=45):
    # 生成一條水平線的核
    psf = np.zeros(img_size, dtype=np.float32)
    center = (img_size[0] // 2, img_size[1] // 2)
    psf[center[0], max(center[1] - length // 2, 0):min(center[1] + length // 2, img_size[1])] = 1

    # plt.figure()
    # plt.imshow(psf, cmap='gray')
    # plt.title('Motion Blur PSF')
    # plt.axis('off')
    # plt.savefig('motion_blur_psf.png')
    # plt.show()

    # 生成旋轉矩陣，對 PSF 進行旋轉
    rotation_matrix = cv2.getRotationMatrix2D((center[1],center[0]), angle, 1)
    dsize = (img_size[1], img_size[0])
    psf = cv2.warpAffine(psf, rotation_matrix, dsize, flags=cv2.INTER_CUBIC)

    # 正規化 PSF，使其總和為 1
    if psf.sum() != 0:
        psf /= psf.sum()

    # 顯示生成的 PSF
    plt.figure()
    plt.imshow(psf, cmap='gray')
    plt.title('Motion Blur PSF')
    plt.axis('off')
    plt.savefig('motion_blur_psf.png')
    # plt.show()

    return psf

"""
TODO Part 2: Wiener filtering
"""
# 參考: https://blog.csdn.net/wsp_1138886114/article/details/95024180
def wiener_filtering(img, psf, K=0.01):
    # 拆分 BGR 通道
    b, g, r = cv2.split(img)
    output_img = []

    # 將 psf 從線在中間的pattern，移動成線在左上右下的pattern
    psf = np.fft.fftshift(psf)

    # plt.figure()
    # plt.imshow(psf, cmap='gray')
    # plt.title('shift of Motion Blur PSF')
    # plt.axis('off')
    # plt.savefig('shift of Motion Blur PSF.png')
    # plt.show()

    # 對 PSF 進行傅立葉變換
    psf_fft = np.fft.fft2(psf)

    psf_fft_magnitude = np.log(np.abs(psf_fft) + 1)  # 取對數，防止動態範圍過大
    plt.figure()
    plt.imshow(psf_fft_magnitude, cmap='gray')
    plt.title('Fourier Transform of Motion Blur PSF')
    plt.axis('off')
    plt.savefig('Fourier Transform of Motion Blur PSF.png')
    # plt.show()

    ### 對每個通道進行 Wiener 濾波
    for channel in [b, g, r]:
        # 將影像進行傅里葉變換
        channel_fft = np.fft.fft2(channel)
        
        # 計算濾波器 H* / (|H|^2 + K)
        psf_fft_1 = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + K)
        
        # 進行濾波並轉換回空間域
        result_fft = channel_fft * psf_fft_1

        # 逆傅里葉變換到空間域
        result = np.fft.ifft2(result_fft)
        result = np.abs(result)   # result = np.abs(result)

        # 將結果剪裁到 [0, 255] 並轉換為 uint8
        result = np.clip(result, 0, 255).astype('uint8')
        output_img.append(result)

    # 合併三個通道
    output_img = cv2.merge([output_img[0], output_img[1], output_img[2]])
    return output_img


def compute_PSNR(image_original, image_restored):
    # PSNR = 10 * log10(max_pixel^2 / MSE)
    psnr = 10 * np.log10(255 ** 2 / np.mean((image_original.astype(np.float64) - image_restored.astype(np.float64)) ** 2))

    return psnr


"""
Main function
"""
def main():
    # img_blurred = cv2.imread("data/image_restoration/testcase1/input_blurred.png")
    # psf = generate_motion_blur_psf(img_blurred.shape[:2])
    # gap = np.ones((img_blurred.shape[0], 2, 3), dtype=np.uint8) * 255  # 全白空隙
    # constrained_least_square_img1 = constrained_least_square_filtering(img_blurred, psf, 0.8)
    # constrained_least_square_img2 = constrained_least_square_filtering(img_blurred, psf, 5)
    # cv2.imshow("window", np.hstack([constrained_least_square_img1, gap, constrained_least_square_img2]))
    # cv2.imwrite("part3_2.jpg", np.hstack([constrained_least_square_img1, gap, constrained_least_square_img2]))
    # cv2.waitKey(0)
    # return
    for i in range(2):
        img_original = cv2.imread("data/image_restoration/testcase{}/input_original.png".format(i + 1))
        img_blurred = cv2.imread("data/image_restoration/testcase{}/input_blurred.png".format(i + 1))

        # TODO Part 1: Motion blur PSF generation
        psf = generate_motion_blur_psf(img_original.shape[:2], 40, 45)

        # TODO Part 2: Wiener filtering
        wiener_img = wiener_filtering(img_blurred, psf)
        cv2.imwrite("wiener_img{}.jpg".format(i+1), wiener_img)

        print("\n---------- Testcase {} ----------".format(i))
        print("Method: Wiener filtering")
        print("PSNR = {}\n".format(compute_PSNR(img_original, wiener_img)))
        cv2.imwrite("wiener_img{}(ori).jpg".format(i+1), np.hstack([img_blurred, wiener_img]))
        # cv2.imshow("window", np.hstack([img_blurred, wiener_img]))
        #         
        # cv2.imshow("window", np.hstack([img_blurred, wiener_img, constrained_least_square_img,inverse_img]))
        cv2.imwrite("res_all{}.jpg".format(i+1), np.hstack([img_blurred, wiener_img, constrained_least_square_img]))
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
