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
    :return: gray_sale np array that elements are in range [0, 255]
    """
    image_array = np.array(image, dtype=np.float32)
    image_array /= 255
    R, G, B = image_array[..., 0], image_array[..., 1], image_array[..., 2]
    gray_scale = 0.3 * R + 0.59 * G + 0.11 * B
    return np.array(gray_scale * 255, dtype=np.uint8)


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
    return np.unique(image_array, return_counts=True)


def create_mapper(value, frequency, image_height, image_width):
    """Creates a mapper that is a dictionary to map each value intensity of the image to a new intensity"""
    cumulative_frequency = np.cumsum(frequency)
    color_levels = value.shape[0]

    mapper = {}
    for color, cum_sum in zip(value, cumulative_frequency):
        mapper[color] = np.ceil((color_levels - 1) * cum_sum / (image_width * image_height))

    return mapper


def map_image(image_array, mapper):
    """Map intensity values of the image_array into new values based on the mapper"""
    image_height, image_width = image_array.shape
    for i in range(image_height):
        for j in range(image_width):
            image_array[i, j] = mapper[image_array[i, j]]


def store_image(image_array):
    """Stores the given image_array"""
    image = Image.fromarray(np.array(image_array, dtype=np.uint8))
    image.save('out.jpg')


def main():
    image_path = "image.png"

    image_array = read_grayscale_image(image_path)
    image_height, image_width = image_array.shape

    value, frequency = get_value_frequencies(image_array)
    draw_histogram(value, frequency)

    mapper = create_mapper(value, frequency, image_height, image_width)
    map_image(image_array, mapper)

    store_image(image_array)


if __name__ == '__main__':
    main()
