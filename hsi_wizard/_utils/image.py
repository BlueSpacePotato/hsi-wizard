"""

todo:   Float or Int handler
        Levels
        Curves
        Sepia
        Color balance (including white balance)
        Vibrance
        Exposure
        Sharpening

"""
from wizard import DataCube
import numpy as np

from matplotlib import pyplot as plt
from skimage.filters import meijering, sato, frangi, hessian
from wizard.utils.decorators import check_limits


def extend_image(img: np.array, extend_x: int, extend_y: int) -> np.array:
    """
    extend image by 2 * extend_x and 2 * extend_y
    extend_x/y are adding zero borders in the given size

    :param img:
    :param extend_x:
    :param extend_y:
    :return: extended_image
    :rtype: np.array
    """

    x, y, z = img.shape

    new_img = np.zeros(shape=(x + extend_x*2, y + extend_y*2, z), dtype=img.dtype)
    new_img[extend_x:-extend_x, extend_y:-extend_y] = img

    return new_img


def decrease_image(img: np.array, decrease_x: int, decrease_y: int) -> np.array:
    """
    extend image by 2 * extend_x and 2 * extend_y
    extend_x/y are adding zero borders in the given size

    :param img:
    :param decrease_x:
    :param decrease_y:
    :return: extended_image
    :rtype: np.array
    """
    return img[decrease_x:-decrease_x, decrease_y:-decrease_y]


def get_output_size(image_lengths:int, filter_lengths:int, stride:int) -> [int, bool]:
    """
    Calc length of feature map. The functions should return an integer value, if the math is not possible, the function
    returns an 0.
    
    :param image_lengths: len of one image side
    :type image_lengths: int
    :param filter_lengths:  len of one the filter size
    :type filter_lengths: int
    :param stride: filter stride
    :type stride: int
    ...
    :raisees None:
    ...
    :return: feature_lengths as int, if the math is not possible return 0
    :rtype: int

    >>> get_output_size(10, 2, 1)
    9

    >>> get_output_size(10, 3, 2)
    0

    """
    feature_lengths = (image_lengths - filter_lengths) / stride + 1
    return int(feature_lengths) if feature_lengths.is_integer() else 0  # check if if_integer works, otherwise math.isclose()


def feature_map(img: np.array, filter: np.array, padding:str='const', stride_x: int = 1, stride_y: int = 1):
    """
    Generating a feature map, by stepping over the image with a given filter function.

    :param img:
    :param filter:
    :param padding:
    :param step_x:
    :param step_y:
    :return:
    """

    if stride_x == 0:
        raise ValueError('step_x cant be 0 (zero)')
    elif stride_y == 0:
        raise ValueError('step_y cant be 0 (zero)')
    
    feature_map_len_x = get_output_size(img.shape[0], filter.shape[0], stride_x)
    feature_map_len_y = get_output_size(img.shape[1], filter.shape[1], stride_y)

    if feature_map_len_x == 0:
        print('x is wong')
        while feature_map_len_x == 0:
            stride_x += 1
            feature_map_len_x = get_output_size(img.shape[0], filter.shape[0], stride_x)
        print(f'stride x {stride_x} whould work')
        return None
    if feature_map_len_y == 0:
        print('y is wrong')
        while feature_map_len_y == 0:
            stride_y += 1
            feature_map_len_y = get_output_size(img.shape[1], filter.shape[1], stride_y)
        print(f'stride y {stride_y} whould work')
        return None

    feature_img = np.zeros(
        shape=(
            feature_map_len_x,
            feature_map_len_y,
            img.shape[2]
        )
    )

    for x in range(feature_map_len_x):
        x1 = x * stride_x
        x2 = x * stride_x + filter.shape[0]
        for y in range(feature_map_len_y):
            y1 = y * stride_y
            y2 = y * stride_y + filter.shape[1]
            mini_img = img[x1:x2, y1:y2]
            pixel = np.sum(mini_img * filter)
            feature_img[x, y] = pixel

    feature_img = feature_img / feature_img.max()

    return feature_img


@check_limits
def rgb_to_grayscale_average(image: np.array) -> np.array:
    """
    rgb to grayscale with average values

    color/3 to avoid conflicts with uint8 images

    :param image:
    :return:
    """
    return image[:, :, 0] / 3 + image[:, :, 1] / 3 + image[:, :, 2] / 3


@check_limits
def rgb_to_grayscale_wight(image: np.array, r_weight: float = 0.299, g_weight: float = 0.587,
                           b_weight: float = 0.114) -> np.array:
    """

    Parameters
    ----------
    :param image: np.array
        rgb_image as input
    :param r_weight: float
        weight for adjusting red color
    :param g_weight: float
        weight for adjusting green color
    :param b_weight: float
        weight for adjusting for blue value

    :rtype: np.array
    :return: grayscale image

    :raise ValueError: if

    """
    return image[:, :, 0] * r_weight + image[:, :, 1] * g_weight + image[:, :, 2] * b_weight


@check_limits
def brightness(image: np.array, delta: int) -> np.array:
    """
    adjust brightness of an image

    :param image:
    :param delta:
    :return:
    """
    tmp = image + delta
    tmp[tmp < delta] = 255
    return tmp


@check_limits
def contrast(image: np.array, beta: int) -> np.array:
    """
    :source: https://towardsdatascience.com/image-processing-and-pixel-manipulation-photo-filters-5d37a2f992fa

    :param image:
    :param beta:

    :return: image
    :rtype: np.array
    """

    # Calculate average brightness
    u = np.mean(image, axis=2)
    u_mean = u.mean()

    # Calculate factor
    if beta == 255:
        alpha = np.infty
    else:
        alpha = (255 + beta) / (255 - beta)

    image = ((image[:, :] - u_mean) * alpha + u_mean).astype('int')  # todo: avoid clipping values

    return image


@check_limits
def saturation(image, beta):
    """
    todo: avoid clipping

    :param image:
    :param beta:
    :return:
    """

    if beta >= 255:
        alpha = np.infty
    else:
        alpha = (255 + beta) / (255 - beta)

    u = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3

    _image = np.empty_like(image)
    _image[:, :, 0] = alpha * (image[:, :, 0] - u) + u
    _image[:, :, 1] = alpha * (image[:, :, 1] - u) + u
    _image[:, :, 2] = alpha * (image[:, :, 2] - u) + u
    return _image


@check_limits
def gamma_correction(image, gamma) -> np.array:
    """
    gamma correction

    :param image:
    :param gamma:
    :return:
    :rtype: np.array

    """
    return 255 * (image / 255) ** gamma


def floyed_steiberg_dithering(image):
    """
    code inspired by
    :source: https://scipython.com/blog/floyd-steinberg-dithering/
    :by: christian
    :source data:13 Oct 2021

    :param image:
    :return:
    """

    def get_new_val(old_val, nc):
        """
        Get the "closest" colour to old_val in the range [0,1] per channel divided
        into nc values.

        """

        return np.round(old_val * (nc - 1)) / (nc - 1)

    # For RGB images, the following might give better colour-matching.
    # p = np.linspace(0, 1, nc)
    # p = np.array(list(product(p,p,p)))
    # def get_new_val(old_val):
    #    idx = np.argmin(np.sum((old_val[None,:] - p)**2, axis=1))
    #    return p[idx]

    def fs_dither(img, nc):
        """
        Floyd-Steinberg dither the image img into a palette with nc colours per
        channel.

        """

        arr = np.array(img, dtype=float) / 255

        for ir in range(new_height):
            for ic in range(new_width):
                # NB need to copy here for RGB arrays otherwise err will be (0,0,0)!
                old_val = arr[ir, ic].copy()
                new_val = get_new_val(old_val, nc)
                arr[ir, ic] = new_val
                err = old_val - new_val
                # In this simple example, we will just ignore the border pixels.
                if ic < new_width - 1:
                    arr[ir, ic + 1] += err * 7 / 16
                if ir < new_height - 1:
                    if ic > 0:
                        arr[ir + 1, ic - 1] += err * 3 / 16
                    arr[ir + 1, ic] += err * 5 / 16
                    if ic < new_width - 1:
                        arr[ir + 1, ic + 1] += err / 16

        carr = np.array(arr / np.max(arr, axis=(0, 1)) * 255, dtype=np.uint8)
        return Image.fromarray(carr)

    def palette_reduce(img, nc):
        """Simple palette reduction without dithering."""
        arr = np.array(img, dtype=float) / 255
        arr = get_new_val(arr, nc)

        carr = np.array(arr / np.max(arr) * 255, dtype=np.uint8)
        return Image.fromarray(carr)

    for nc in (2, 3, 4, 8, 16):
        print('nc =', nc)
        dim = fs_dither(img, nc)
        dim.save('dimg-{}.jpg'.format(nc))
        rim = palette_reduce(img, nc)
        rim.save('rimg-{}.jpg'.format(nc))


def ridge_filter_overview(img: np.array) -> np.array:

    image = img
    cmap = plt.cm.gray

    kwargs = {'sigmas': [1], 'mode': 'reflect'}

    fig, axes = plt.subplots(2, 4)
    for i, black_ridges in enumerate([1, 0]):
        for j, func in enumerate([meijering, sato, frangi, hessian]):
            kwargs['black_ridges'] = black_ridges
            result = func(image, **kwargs)
            axes[i, j].imshow(result, cmap=cmap, aspect='auto')
            if i == 0:
                axes[i, j].set_title(['Meijering\nneuriteness',
                                  'Sato\ntubeness', 'Frangi\nvesselness',
                                  'Hessian\nvesselness'][j])
            if j == 0:
                axes[i, j].set_ylabel('black_ridges = ' + str(bool(black_ridges)))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()
    plt.show()

def ridge_filter(img: np.array, filter:str='frangi', black_ridges:bool=True) -> np.array:
    """
    'Meijering neuriteness',
    'Sato tubeness', 
    'Frangi vesselness',
    'Hessian vesselness'
    
    :param img:
    :param filter: 
    :param black_ridges: 
    :return: 
    """

    image = img
    cmap = plt.cm.gray

    kwargs = {'sigmas': [1], 'mode': 'reflect'}
    
    if filter == 'frangi':
        func = frangi
    elif filter == 'meijering':
        func = meijering
    elif filter == 'sato':
        func = sato
    elif filter == 'hessian':
        func = hessian
    else:
        raise ValueError(f'Filter `{filter} is unknown.`')

    kwargs['black_ridges'] = black_ridges
    
    return func(image, **kwargs)
    




def hessian_filter(data) -> np.array:

    if type(data) is DataCube:
        img = np.transpose(data.cube, (2,1,0))
        print(img.shape)
    else:
        img = data
    
    kwargs = {'sigmas': [1], 'mode': 'reflect'}
    hessian_img = hessian(img, **kwargs)

    return hessian_img



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import wizard
    import os
    from wizard.utils.fileHandler import read, images_from_folder_to_dc, read_tdms
    from wizard.utils.plotting.plotter import plotter
    data_path = '/home/flx/Documents/Daten/Multi-Image/2022_12_14_MultiImage_S2_A_Mitte'

    raman = wizard.DataCube.read(
        path=os.path.join(
            data_path,
            'R_BD50x_100mW_100µm_100+600µm_-7_5000.0ms.tdms'
        )
    )

    raman = wizard.processing.preset_raman(raman)
    wizard.utils.plotting.plotter.plotter(raman)

    print(raman.shape)

    # img = plt.imread('/home/flx/Documents/wizard-of-os/resources/SampleData/land-rover-1050x670.jpg')
    img =  np.transpose(raman.cube[450:453], (1, 2, 0))
    print(img.shape)
    img = img - img.min()
    img = img / img.max()

    plt.imshow(img)
    plt.title('img')
    plt.show()

    gray = rgb_to_grayscale_wight(img)
    plt.imshow(gray, cmap='gray')
    plt.title('gray - wight')
    plt.show()

    gray = rgb_to_grayscale_average(img)
    plt.title('img - average')
    plt.imshow(gray, cmap='gray')
    plt.show()

    bright = brightness(img, delta=0.20)
    plt.imshow(bright)
    plt.title('+ brightness')
    plt.show()

    bright = brightness(img, delta=-0.2)
    plt.imshow(bright)
    plt.title('- brightness')
    plt.show()

    cont = contrast(img, beta=50)
    plt.imshow(cont)
    plt.title('+ contrast')
    plt.show()

    cont = contrast(img, beta=-50)
    plt.imshow(cont)
    plt.title('- contrast')
    plt.show()

    sat = saturation(img, beta=0)
    plt.imshow(sat)
    plt.title('+ saturation')
    plt.show()

    sat = saturation(img, beta=-0)
    plt.imshow(sat)
    plt.title('- saturation')
    plt.show()

    gam = gamma_correction(img, gamma=0.9)
    plt.imshow(gam)
    plt.title('+ gamma')
    plt.show()

    gam = gamma_correction(img, gamma=1.1)
    plt.imshow(gam)
    plt.title('- gamma')
    plt.show()

    ridge_filter(img)
  
    tmp_img = extend_image(img, 20, 10)
    plt.imshow(tmp_img)
    plt.show()

    plt.imshow(decrease_image(tmp_img, 20, 10))
    plt.show()

    filter = np.zeros(shape=(54, 54, 3))
    filter[:, 26, :] = 1

    f_map_1 = feature_map(img, filter=filter, stride_x=7, stride_y=17)
    plt.imshow(f_map_1)
    plt.show()

    filter = np.zeros(shape=(54, 54, 3))
    filter[26, :, :] = 1

    f_map_2 = feature_map(img, filter=filter, stride_x=7, stride_y=17)
    plt.imshow(f_map_2)
    plt.show()

    plt.imshow((f_map_1+f_map_2)/2)
    plt.show()
    
    from filter import get_gaussian
    
    kernel = get_gaussian()
    filter = np.array((kernel, kernel, kernel))
    filter = np.transpose(filter, (2, 1, 0))
    print(filter.shape)
    f_map_3 = feature_map(img, filter=filter, stride_x=7, stride_y=17)
    plt.imshow(f_map_3)
    plt.show()
