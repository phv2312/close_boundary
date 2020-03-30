import os
import cv2
import csv
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from multiprocessing import Pool, cpu_count


def process_CA_for_evaluation(ca_csv_path):
    # Read from CA.
    with open(ca_csv_path) as csv_file:
        ca = list(csv.reader(csv_file, delimiter=','))
    csv_file.close()

    # Turn CA list into a dict for easier processing.
    ca_dict = {}
    for i, row in enumerate(ca[1:]):
        img_name, _, _, _, n_contours, row1, col1, row2, col2 = row
        # Turn points to strings to make a dict.
        coor1 = '%s_%s' % (row1, col1)
        coor2 = '%s_%s' % (row2, col2)
        row_index = i + 1
        if img_name in ca_dict:
            ca_dict[img_name]['coor_info'].append((coor1, coor2, row_index))
        else:
            ca_dict[img_name] = {'n_contours': int(n_contours), 'coor_info': [(coor1, coor2, row_index)]}
    return ca_dict


def process_CA_for_preprocessing(ca_csv_path):
    # Read from CA.
    with open(ca_csv_path) as csv_file:
        ca = list(csv.reader(csv_file, delimiter=','))
    csv_file.close()
    # Turn CA list into a dict for easier processing.
    ca_dict = {}
    for i, row in enumerate(ca[1:]):
        img_name, cut_name, reference_name = row[:3]
        reference_name = reference_name[:-4] + '.tga'
        ca_dict[img_name] = reference_name
    return ca_dict


def cv2_imgshow(image):
    plt.imshow(image)
    plt.show()


def run_multi_process(f, args, n_cpu = -1):
    if n_cpu == -1: n_cpu = cpu_count()

    print(n_cpu)
    p = Pool(n_cpu)
    output = p.map(f, args)

    return output


def find_bound(ref_colored_img, start, stop, step):
    """Finds the bound within the interval [start, stop) by scanning from the 'start' row to the 'stop' row in the
    ref_colored_img array. One bound is found when the sum of all of its pixels becomes larger than 0.

    Args:
        ref_colored_img (numpy array): reference colored image
        start (int): the starting row to be scanned
        stop (int): the row at which the scan stops
        step (int): specifies the incrementation

    Returns:
        int: the upper/lower bound. If the bound does not exist, return 'start'.
    """

    # Scan from 'start' to 'stop', stop the scan and draw the line at the row whose sum of all pixels are larger than 0.
    for i in range(start, stop, step):
        if np.sum(ref_colored_img[i]) != 0:
            return i
    return start


def crop_image(img_lst, upper_bound, lower_bound):
    """Crops all the images in the image list.

    Args:
        img_lst (numpy array list): list of image arrays to be cropped
        upper_bound (int): the starting row of the cropped image
        lower_bound (int): the end row of the cropped image

    Returns:
        numpy array list: list of all the cropped images
    """
    crop_list = []
    for img in img_lst:
        crop_list.append(img[upper_bound:lower_bound].copy())
    return crop_list


def preprocess_imgs(sketch, ref_colored_img, diff=70):
    """Crops all the images in img_list. The bounds of the cropped image can be found from the reference colored image.

    Args:
        sketch (numpy array): 2-D sketch array
        ref_colored_img(numpy array): 3_D reference colored image array
        diff (int): the upper bound is determined within (0, diff) rows, lower bound within (height - diff, height).
                    If the bound does not lie within the interval, no line will be drawn.

    Returns:
        numpy array list: list of processed images
        tuple (int, int): (upper_bound, lower_bound)
    """
    assert sketch.shape[0] == ref_colored_img.shape[0] \
           and sketch.shape[1] == ref_colored_img.shape[1], 'Incompatible size'
    height = ref_colored_img.shape[0]
    ref_colored_img = 255 - ref_colored_img
    upper_bound = find_bound(ref_colored_img, 0, diff, 1)
    lower_bound = find_bound(ref_colored_img, height - 1, height - diff, -1) + 1
    preprocessed_img = crop_image([sketch], upper_bound, lower_bound)
    return preprocessed_img, (upper_bound, lower_bound)


def pad_img(colored_sketch, bound, original_shape):
    """Pads the sketch so that its shape equals the original shape.

    Args:
        colored_sketch (numpy array): image array (RGB)
        bound (tuple (int, int)): (upper_bound, lower_bound)
        original_shape(tuple): (height, width, color_channels)

    Returns:
        numpy array: padded image (RGB)
    """
    upper_bound, lower_bound = bound
    new_sketch = np.ones((original_shape), np.uint8) * 255
    new_sketch[upper_bound:lower_bound] = colored_sketch
    return new_sketch


def check_filenames(img_paths, sketch_paths):
    """Raise AssertionError if any sketch path does not have the same basename as its corresponding image path.

    Args:
        img_paths (list): list of image paths
        sketch_paths (list): list of sketch paths
    """
    for i, img_path in enumerate(img_paths):
        img_fn = os.path.basename(img_path)
        sketch_fn = os.path.basename(sketch_paths[i])
        assert img_fn == sketch_fn, 'Sketch and image folders contain differently named pair.'


def make_new_folder(folder_fn):
    """Create a new folder. If the folder already exists, delete it and create a new one."""
    if os.path.isdir(folder_fn):
        shutil.rmtree(folder_fn)
    os.makedirs(folder_fn)


def imread(img_path, grayscale=False):
    """Load the image from img_path.

    Args:
        img_path (str): path to the image file
        grayscale (bool): True if the image should be in grayscale, False otherwise

    Returns:
        numpy array: the image array
    """
    if grayscale:
        return np.asarray(Image.open(img_path).convert('L'))
    img = np.asarray(Image.open(img_path))
    return img[:, :, :3]


def imsave(result, output_dir, img_name):
    """Save the image.

    Args:
        result (numpy array): image array to be saved
        output_dir (str): path to output folder
        img_name (str): basename of the image (e.g. output_1.png)
    """
    Image.fromarray(result).save('%s/%s' % (output_dir, img_name))


def check_dimensions(arr_list):
    """Check if all image arrays have the identical heights and widths.

    Args:
        arr_list (numpy array list): list of image arrays

    Returns:
        bool: True if all arrays have identical heights and widths, False otherwise
    """
    shape = arr_list[0].shape
    for arr in arr_list[1:]:
        if arr.shape[0] != shape[0] or arr.shape[1] != shape[1]:
            return False
        shape = arr.shape
    return True


def evaluate_per_pixel(result, label):
    """Calculate the percentage of pixels that are correctly colorized.

    Args:
        result (numpy array): colored sketch (RGB)
        label (numpy array): label (RGB)

    Returns:
        float: accuracy of the colorization process
    """
    height, width, depth = result.shape
    diff = result - label
    reduced_diff = diff[:, :, 0] + diff[:, :, 1] + diff[:, :, 2]
    n_accurate_pixels = height * width - np.count_nonzero(reduced_diff)
    total_pixels = height * width
    accuracy = n_accurate_pixels * 1.0 / total_pixels
    return accuracy, n_accurate_pixels, total_pixels


def color_for_debugging(paired_components, ref_sketch_components, sketch_components, original_shape, write_text=True):
    """Color the components and label each component with its index.

    Args:
        paired_components (dict): type dict, with key/value is the id of component/reference_component.
        ref_sketch_components (dict): dictionary of reference components in the form {index: dict of components' properties}
        sketch_components (dict): dictionary of all sketch components in the form {index: dict of components' properties}
        original_shape (tuple): shape
        write_text (bool): True if the component should be labelled with its index for debugging purposes

    Returns:
        rgb_ref_mask (numpy array): colorized rgb reference mask.
        rgb_mask (numpy array): colorized rgb mask.

    """
    h, w = original_shape[:2]
    chosen_colors = []

    rgb_ref_mask = np.ones(shape=(h, w, 3), dtype=np.uint8) * 255  # cv2.cvtColor(ref_mask, cv2.COLOR_GRAY2RGB)
    rgb_mask = np.ones(shape=(h, w, 3), dtype=np.uint8) * 255  # cv2.cvtColor(_mask, cv2.COLOR_GRAY2RGB)

    ref_colors = {}

    for _id, ref_id in paired_components.items():
        ref_img = ref_sketch_components[ref_id]
        _img = sketch_components[_id]

        # Find the unique value for each label.
        if ref_id not in ref_colors:
            cors_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            while cors_color in chosen_colors:
                cors_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

            ref_colors[ref_id] = cors_color
        else:
            cors_color = ref_colors[ref_id]

        chosen_colors += [cors_color]

        rgb_ref_mask[ref_img['coords'][:, 0], ref_img['coords'][:, 1], :] = cors_color
        rgb_mask[_img['coords'][:, 0], _img['coords'][:, 1], :] = cors_color

    # write text
    if write_text:
        for _id, comp in ref_sketch_components.items():
            cen_y, cen_x = comp['centroid']
            cv2.putText(rgb_ref_mask, str(_id), (int(cen_x), int(cen_y)), cv2.FONT_HERSHEY_COMPLEX, .6,
                        color=(0, 0, 0), thickness=1)

        for _id, comp in sketch_components.items():
            cen_y, cen_x = comp['centroid']
            cv2.putText(rgb_mask, str(_id), (int(cen_x), int(cen_y)), cv2.FONT_HERSHEY_COMPLEX, .6,
                        color=(0, 0, 0), thickness=1)

    return rgb_ref_mask, rgb_mask


def determine_kernel_size(position, endpoint, diff=3):
    """Ensures that the kernel is within margins

    Args:
        position (int): the current row or column of the pixel
        endpoint (int): number of rows or columns of the original image
        diff (int): the width and height of the kernel from the current position

    Returns:
        int, int: the starting row/col and the end row/col of the kernel
    """
    start = position - diff if position >= diff else position
    end = position + diff if position <= endpoint - diff else position
    return start, end


def process_dir_path(dir_path):
    temp = dir_path.split('\\')
    result = temp[0]
    for i in range(1, len(temp), 1):
        result += '/' + temp[i]
    return result


def get_current_sketch_index(n_frames, ref_index, switch_left):
    # If the reference sketch is a middle frame, colorize the frames on the right first and then those on the left.
    # If the reference sketch is the last frame, colorize the frames on the left
    if ref_index < n_frames - 1:
        cur_index = ref_index + 1
    elif ref_index == n_frames - 1:
        cur_index = ref_index - 1
        # Update the switch flag to indicate that the process is moving to the left
        switch_left = True
    else:
        print('reference sketch index %d is out of bound' % ref_index)
        cur_index = -1
    return cur_index, switch_left


def create_a_folder_tree(output_dir, name_list):
    for name in name_list:
        output_dir = output_dir + '/' + name
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def convert_to_visualization_point_format(point):
    row, col = point.split('_')
    return int(col), int(row)


def visualize_debug_img(debug_img_path, results, mode='evaluation', output_dir=None):
    img = cv2.imread(debug_img_path)
    if mode is 'evaluation':
        for start, end in results.items():
            start = convert_to_visualization_point_format(start)
            end = convert_to_visualization_point_format(end)
            cv2.line(img, start, end, (255, 0, 0), 2)
    else:
        for point_pair in results:
            start, end = point_pair
            start = (start[1], start[0])
            end = (end[1], end[0])
            cv2.line(img, start, end, (255, 0, 0), 2)

    if output_dir:
        output_path = '%s/%s' % (output_dir, os.path.basename(debug_img_path))
        cv2.imwrite(output_path, img)
    return img

