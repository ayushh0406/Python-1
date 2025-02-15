import math
import cv2
import numpy as np


def Representational(r, g, b):
    """
    Convert an RGB pixel to a grayscale equivalent using the ITU-R BT.601 formula.
    """
    return 0.299 * r + 0.287 * g + 0.114 * b


def calculate(img):
    """
    Calculate the grayscale equivalent of an image.
    """
    b, g, r = cv2.split(img)
    pixelAt = Representational(r, g, b)
    return pixelAt


def add_noise(img):
    """
    Add random Gaussian noise to an image.
    """
    row, col, ch = img.shape
    mean = 0
    sigma = 0.1 * 255
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = img + gauss.reshape(row, col, ch)
    return noisy


def display_images(original, compressed, noisy):
    """
    Display the original, compressed, and noisy images side by side.
    """
    concatenated = np.hstack((original, compressed, noisy))
    cv2.imshow('Original vs. Compressed vs. Noisy', concatenated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculate_psnr(original, compressed):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    height, width = original.shape[:2]

    original_pixel_at = calculate(original)
    compressed_pixel_at = calculate(compressed)

    diff = original_pixel_at - compressed_pixel_at
    error = np.sum(np.abs(diff) ** 2) / (height * width)

    PSNR = -(10 * math.log10(error / (255 * 255)))
    return PSNR


def main():
    # Loading images (original image and compressed image)
    original_image = cv2.imread("original_image.png", 1)
    compressed_image = cv2.imread("compressed_image.png", 1)

    # Adding noise to the compressed image
    noisy_image = add_noise(compressed_image)

    # Calculating PSNR values
    psnr_value = calculate_psnr(original_image, compressed_image)
    psnr_noisy_value = calculate_psnr(original_image, noisy_image)

    # Displaying the images
    display_images(original_image, compressed_image, noisy_image)

    # Printing PSNR values
    print("PSNR value between original and compressed image: {:.2f}".format(psnr_value))
    print("PSNR value between original and noisy image: {:.2f}".format(psnr_noisy_value))


if __name__ == "__main__":
    main()

