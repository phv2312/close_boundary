import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line
from skimage.morphology import thin

def imgshow(im):
    plt.imshow(im)
    plt.show()

def do_thin(cv_im, do_preprocess=False):
    """
    Thinning the given image to 1-pixel boundary
    :param cv_im: np.ndarray, should be binarized to better performance
    :return:
    """
    cv_im = cv2.threshold(cv_im, 254, 255, cv2.THRESH_BINARY)[1]

    if do_preprocess:
        cv_im = cv2.GaussianBlur(cv_im, (7, 7), 0)
        cv_im = cv2.threshold(cv_im, 254, 255, cv2.THRESH_BINARY)[1]

    cv_im[cv_im == 255] = 1
    cv_im = 1 - cv_im

    thinned_im = thin(cv_im).astype(np.uint8) * 255
    return thinned_im

def _count_active_neighbor(row, col, matrix):
    """
    Count the active (the boundary pixel, has value > 0) neighbor pixels of the given pixel.
    :param row: int
    :param col: int
    :param matrix: np.ndarray
    :return:
    """
    h, w = matrix.shape

    # get the neighbor ids
    min_row = max(row - 1, 0)
    max_row = min(row + 1 + 1, h)

    min_col = max(col - 1, 0)
    max_col = min(col + 1 + 1, w)

    coords = []
    for _row in range(min_row, max_row):
        for _col in range(min_col, max_col):
            if (_row, _col) != (row, col) and matrix[_row, _col] > 0:
                coords += [(_row, _col)]

    return coords, matrix[min_row:max_row, min_col:max_col]

def to_neighbor_matrix(cv2_im):
    """
    :param cv2_im: 255 is boundary, 0 is background. cv2_im is a binary matrix
    :return: a matrix whose element represent the number of active pixels (pixel > 0).
    """
    rows, cols = np.where(cv2_im != 0)
    result = np.zeros_like(cv2_im, dtype=np.uint8)
    for row, col in zip(rows, cols):
        neighbors, _matrix = _count_active_neighbor(row, col, matrix=cv2_im)
        result[row, col] = len(neighbors)

        if False:
            print("row:%d-col:%d has %d neighbors..." % (row, col, len(neighbors)))

    return result

def calc_gradient(M):
    """
    Calculating the angle of point ...
    :param M:
    :return:
    """
    sobel_x = cv2.Sobel(M, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(M, cv2.CV_64F, 0, 1, ksize=3)

    theta = np.arctan2(sobel_y, sobel_x)
    return theta

def find_direction(img, rows, cols, D):
    """
    Mapping angle to the next neighbor points ...
    """
    angle = D * 180. / np.pi
    angle[angle < 0] += 360

    #
    directions = [(1,0), (1,1), (0,1), (-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0)]
    angles = np.array([0,45,90,135,180,225,270,315,360])

    #
    result_dct = {}
    for i, j in zip(rows, cols):
        _d = angle[i, j]

        _tmp = np.abs(angles - _d)
        _min_id = np.argmin(_tmp)

        if _min_id == len(angles) - 1: _min_id = 2

        _next = directions[_min_id]
        #img[i + _next[1], j + _next[0]] = 127

        result_dct[(i,j)] = _next

    return result_dct

def get_neighbor_ids(_row, _col, _cv2_im, only_active_pixel=True, max_neighbor=1):
    """

    :param _row: ez
    :param _col: ez
    :param _cv2_im: ez
    :param only_active_pixel: if True, get only active pixel (pixel > 0) else get full
    :return:
    """
    h, w = _cv2_im.shape

    min_row = max(_row - max_neighbor, 0)
    max_row = min(_row + max_neighbor + 1, h)

    min_col = max(_col - max_neighbor, 0)
    max_col = min(_col + max_neighbor + 1, w)

    coords = []
    for __row in range(min_row, max_row):
        for __col in range(min_col, max_col):

            if (__row, __col) != (_row, _col):
                if not only_active_pixel:
                    coords += [(__row, __col)]
                else:
                    if _cv2_im[__row, __col] > 0:
                        coords += [(__row, __col)]

    return coords

def is_intersection_point(row, col, cv2_im, max_traveled_pixel=10):
    """
    One assumption is that the un-clo10ed boundary is just a mistake from the client.
    So that it can not be the intersection point of two lines.

    > Check if as the pixel (row, col) is the intersection point of two lines.
    :param row:
    :param col:
    :param cv2_im:1
    :return:
    """

    stack = [(row, col)]
    traveled = []
    tmp_max_traveled_pixel = max_traveled_pixel
    while (len(stack) > 0 and tmp_max_traveled_pixel > 0):
        cur_row, cur_col = stack.pop(0)
        traveled += [(cur_row, cur_col)]

        neighbor_coords = get_neighbor_ids(cur_row, cur_col, cv2_im)
        neighbor_coords = [coord for coord in neighbor_coords if coord not in traveled + stack]

        n_neighbor = len(neighbor_coords)
        if False:
            print("row: %d,col: %d has %d neighbor, %d" % (cur_row, cur_col, n_neighbor, tmp_max_traveled_pixel))

        if n_neighbor < 1:
            pass
        elif n_neighbor == 1:
            stack += neighbor_coords
            tmp_max_traveled_pixel -= 1
        else:
            return True

    return False

def choose_pair_by_distance(rows, cols, max_distance, return_matrix = False):
    """
    Choose the pair for each key_point (r,c) by compare the distance between them.
    :param rows: list of row
    :param cols: list of column
    :param max_distance: max distance to be considered as pair or not
    :return:
    """
    coords = np.array([[row, col] for (row, col) in zip(rows, cols)], dtype=np.int32)  # (n_samples,2)

    # sorted by norm2
    distance = np.expand_dims(coords, axis=0) - np.expand_dims(coords, axis=1)  # (n_samples, n_samples, 2)
    distance = np.linalg.norm(distance, ord=2, axis=-1).T  # (n_samples, n_samples)
    distance[np.arange(len(rows)), np.arange(len(rows))] = np.inf

    # get the min distance
    min_ids = np.argmin(distance, axis=-1) # n_samples,

    pair = {k:v for k,v in enumerate(min_ids) if distance[k,v] <= max_distance}
    if return_matrix == False:
        return pair
    else:
        return pair, distance

def normalize_sub_im(point1, point2, cv2_im, padding, org_shape):
    h, w = org_shape

    point1 = np.array(point1) # r, c
    point2 = np.array(point2) # r, c

    # get min, max
    min_h, min_w = min([point1[0], point2[0]]), min([point1[1], point2[1]])
    max_h, max_w = max([point1[0], point2[0]]), max([point1[1], point2[1]])

    # add padding
    min_h, min_w = max(min_h - padding, 0), max(min_w - padding, 0)
    max_h, max_w = min(max_h + padding + 1, h), min(max_w + padding + 1, w)

    # apply normalize
    point1 -= [min_h, min_w]
    point2 -= [min_h, min_w]
    sub_cv2_im = cv2_im[min_h:max_h, min_w:max_w]

    return point1, point2, sub_cv2_im

def do_exist_path_btw_points(point1, point2, cv2_im, padding=2):
    """
    Check if exist path from point1 to point2.
    :param point1: (r1,c1)
    :param point2: (r2,c2)
    :param cv2_im:
    :return:
    """
    point1, point2, cv2_im = normalize_sub_im(point1, point2, cv2_im, padding, cv2_im.shape)

    # running algorithm
    stack = [point1]
    traveled = []
    while (len(stack) > 0):
        cur_row, cur_col = stack.pop(0)
        traveled += [(cur_row, cur_col)]

        neighbor_coords = get_neighbor_ids(cur_row, cur_col, cv2_im)
        neighbor_coords = [coord for coord in neighbor_coords if coord not in traveled + stack]

        for (_row, _col) in neighbor_coords:
            if (_row, _col) == tuple(point2):
                return True
            else:
                stack += [(_row, _col)]

    return False

def connect_keypoint_to_boundary(point1, cv2_im, max_distance=6, direction_dct = None):
    h, w = cv2_im.shape

    n_iter = max_distance
    r, c = point1
    direction = direction_dct[(r,c)][::-1]

    next_rs, next_cs = [r + _ * direction[0] for _ in range(1, n_iter + 1)], [c + _ * direction[1] for _ in range(1, n_iter + 1)]
    next_pxls = [cv2_im[_r, _c] if (0 <= _r < h and 0 <= _c < w) else 0 for _r, _c in zip(next_rs, next_cs)]

    if 255 not in next_pxls: return -1, -1
    else:
        _id = next_pxls.index(255)

        return next_rs[_id], next_cs[_id]

def can_connect_two_points(point1, point2, cv2_im):
    def is_in_neighbor(r, c, org_r, orc_c, nb_k):
        if 0 <= np.abs(r - org_r) <= nb_k and 0 <= np.abs(c - orc_c) <= nb_k:
            return True
        return False

    line_rs, line_cs = line(*point1, *point2)

    traveled = []
    for r, c in zip(line_rs, line_cs):
        nb_points = get_neighbor_ids(r, c, cv2_im, only_active_pixel=True)

        if is_in_neighbor(r, c, point1[0], point1[1], 1): continue
        if is_in_neighbor(r, c, point2[0], point2[1], 1): continue

        traveled += [(r,c)]
        nb_points = [point for point in nb_points if point not in traveled]
        if len(nb_points) > 1: return False

    return True

def match_direction(point1, point2, direction_dct, org_shape):
    p1_r, p1_c = point1
    p2_r, p2_c = point2
    h, w = org_shape

    def set(y,x,h,w,im):
        if x <= 0: return
        if y <= 0: return

        if y >= h: return
        if x >= w: return

        im[y,x] = 255

    d1 = direction_dct[(p1_r, p1_c)][::-1] # (y,x)
    d2 = direction_dct[(p2_r, p2_c)][::-1] # (y,x)

    mat = np.zeros(shape=(h,w),dtype=np.uint8)

    iter = 150
    nb_iter = 1
    for _iter in range(iter):
        set(p1_r + _iter * d1[0], p1_c + _iter * d1[1], h, w, mat)
        set(p2_r + _iter * d2[0], p2_c + _iter * d2[1], h, w, mat)

        p1_nbs = get_neighbor_ids(p1_r, p1_c, mat, only_active_pixel=False, max_neighbor=nb_iter)
        p2_nbs = get_neighbor_ids(p2_r, p2_c, mat, only_active_pixel=False, max_neighbor=nb_iter)

        for r, c in p1_nbs:
            set(r + _iter * d1[0], c + _iter * d1[1], h, w, mat)

        for r, c in p2_nbs:
            set(r + _iter * d2[0], c + _iter * d2[1], h, w, mat)

    connect_1 = do_exist_path_btw_points(point1, point2, mat, padding=iter)
    connect_2 = do_exist_path_btw_points(point2, point1, mat, padding=iter)

    is_connect = connect_1 and connect_2
    if is_connect:
        return True

    return False

def convert_output_format(pair_points):
    """Converts the format output pairs to that of evaluation.

    :param pair_points:
    :return:
    """
    result = dict()
    for pair in pair_points:
        start, end = pair
        key = '%d_%d' % (start[0], start[1])
        value = '%d_%d' % (end[0], end[1])
        result[key] = value
    return result
