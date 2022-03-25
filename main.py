from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def read_grayscale_image(image_path):
    """Reads the image using PIL.Image and then converts the image to grayscale"""
    image = Image.open(image_path)
    gray_scale = convert_to_gray_scale(image)

    return gray_scale


def convert_to_gray_scale(image):
    """
    Converts RGB image to grayscale based on the following formula:
    grayscale = ( (0.3 * R) + (0.59 * G) + (0.11 * B) )

    Reference of the formula: https://www.tutorialspoint.com/dip/grayscale_to_rgb_conversion.htm

    :param image: RGB image read by PIL.Image
    :return: gray_sale np array
    """
    image_array = np.array(image, dtype=np.float32)
    image_array /= 255
    R, G, B = image_array[..., 0], image_array[..., 1], image_array[..., 2]
    gray_scale = 0.3 * R + 0.59 * G + 0.11 * B
    return gray_scale


def draw_histogram(value, frequency):
    """
    Draws histogram based on the image intensity values and their corresponding frequencies.
    """

    fig, ax = plt.subplots()
    ax.bar(value, frequency, width=0.8)
    ax.set(title="Image Histogram",
           xlabel="Intensity Value",
           ylabel="Count")
    plt.show()


def get_value_frequencies(image_array):
    """
    Returns a tuple such that the first value of the pair is the unique pixel intensities and the second value of the
    pair is the corresponding frequency.
    """
    image_array = np.array(image_array * 255, dtype=np.uint8)
    return np.unique(image_array, return_counts=True)


def main():
    image_path = "SoroushMehraban.jpg"
    image_array = read_grayscale_image(image_path)

    value, frequency = get_value_frequencies(image_array)
    draw_histogram(value, frequency)


if __name__ == '__main__':
    main()
