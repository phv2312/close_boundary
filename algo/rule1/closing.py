import os

from algo.rule1.utils import *


class ClosingModel:
    def __init__(self, max_traveled_pixel=10, max_pair_distance=10, keypoint_to_boundary_distance=7, DEBUG=True):
        self.max_traveled_pixel = max_traveled_pixel
        self.max_pair_distance = max_pair_distance
        self.keypoint_to_boundary_distance = keypoint_to_boundary_distance

        self.DEBUG = DEBUG
        self.DEBUG_DIR = "debug"
        os.makedirs(self.DEBUG_DIR, exist_ok=True)

    def process(self, tgt_sketch_im, ref_color_im=None):
        pair_points = []
        debug_im = cv2.cvtColor(tgt_sketch_im, cv2.COLOR_GRAY2RGB)

        # thinning, make the boundary width to 1-pixel
        thinned_im = do_thin(tgt_sketch_im)
        if self.DEBUG:
            out_fn = os.path.join(self.DEBUG_DIR, "_1_thin_step1.png")
            cv2.imwrite(out_fn, 255 - thinned_im)

        # the key-points is defined as the pixel which has the most number of neighbor background-pixels.
        neighbor_matrix = to_neighbor_matrix(thinned_im)
        # Get the locations of all the 1's
        rows, cols = np.where(neighbor_matrix == 1)
        if self.DEBUG:
            out_fn = os.path.join(self.DEBUG_DIR, "_2_key_points_step2.png")
            _debug_im = debug_im.copy()
            for row, col in zip(rows, cols):
                cv2.circle(_debug_im, (col, row), radius=1, color=(0, 255, 0), thickness=1)

            cv2.imwrite(out_fn, _debug_im)

        # calculate the direction
        theta = calc_gradient(255 - thinned_im)
        direction_dct = find_direction(thinned_im, rows, cols, theta)  # (x_direction, y_direction)
        if self.DEBUG:
            out_fn = os.path.join(self.DEBUG_DIR, "_3_direction_step3.png")
            _debug_im = debug_im.copy()

            for row, col in zip(rows, cols):
                _next_x, _next_y = direction_dct[(row, col)]
                cv2.arrowedLine(_debug_im, (col, row), (col + _next_x * 6, row + _next_y * 6), color=(0, 255, 0),
                                thickness=1, tipLength=0.5)

            cv2.imwrite(out_fn, _debug_im)

        # post-process the key-point, with our heuristic is that the key-points can not be the intersect btw two lines
        chosen_rows, chosen_cols = [], []
        for row, col in zip(rows, cols):
            if is_intersection_point(row, col, thinned_im, self.max_traveled_pixel): continue

            #
            chosen_rows += [row]
            chosen_cols += [col]

        if self.DEBUG:
            out_fn = os.path.join(self.DEBUG_DIR, "_4_heuristic_intersection_points_step4.png")
            _debug_im = debug_im.copy()
            for row, col in zip(chosen_rows, chosen_cols):
                cv2.circle(_debug_im, (col, row), radius=4, color=(0, 255, 0), thickness=2)

            cv2.imwrite(out_fn, _debug_im)

        """
        Step5: Construct the pair based on the distance btw key-points.
        >> Heuristic, the pair must satisfy following conditions:
            1. Not existing path btw 2 key-points (checking happen in the small window size, not full image)
            2. The line connecting two key-points is not allowed to intersect with other boundary line.
            3. 
        """
        pair = choose_pair_by_distance(chosen_rows, chosen_cols, max_distance=self.max_pair_distance)
        traveled = []
        for src_id, tgt_id in pair.items():
            src_r, src_c = chosen_rows[src_id], chosen_cols[src_id]
            tgt_r, tgt_c = chosen_rows[tgt_id], chosen_cols[tgt_id]

            p1 = (src_r, src_c)
            p2 = (tgt_r, tgt_c)

            if do_exist_path_btw_points(point1=p1, point2=p2, cv2_im=thinned_im.copy(), padding=2):
                continue  # ignore

            can_connect = can_connect_two_points(point1=p1, point2=p2, cv2_im=thinned_im.copy())
            if not can_connect:
                continue  # ignore

            if not match_direction(point1=p1, point2=p2, direction_dct=direction_dct, org_shape=thinned_im.shape):
                continue  # ignore

            traveled += [p1, p2]
            pair_points += [(p1, p2)]

        # print("pair_points")
        # print(pair_points)
        if self.DEBUG:
            out_fn = os.path.join(self.DEBUG_DIR, "_4_heuristic_distance_angle_step4.png")
            _debug_im = debug_im.copy()

            for p1, p2 in pair_points:
                cv2.circle(_debug_im, (p1[1], p1[0]), radius=1, color=(0, 255, 0), thickness=1)
                cv2.circle(_debug_im, (p2[1], p2[0]), radius=1, color=(0, 255, 0), thickness=1)

                cv2.line(_debug_im, (p1[1], p1[0]), (p2[1], p1[0]), color=(0, 0, 255), thickness=2)

            cv2.imwrite(out_fn, _debug_im)

        """
        Step6: Connect the remaining key-points to the nearest boundary (if satisfy the given max distance)
        """
        unset_points = []
        for row, col in zip(chosen_rows, chosen_cols):
            if (row, col) in traveled: continue

            dest_row, dest_col = connect_keypoint_to_boundary(point1=(row, col), cv2_im=thinned_im.copy(),
                                                              max_distance=self.keypoint_to_boundary_distance,
                                                              direction_dct=direction_dct)

            if dest_row != -1:
                pair_points += [((row, col), (dest_row, dest_col))]
            else:
                unset_points += [(row, col)]

        if self.DEBUG:
            out_fn = os.path.join(self.DEBUG_DIR, "_5_final_step5.png")
            _debug_im = debug_im.copy()

            for p1, p2 in pair_points:
                cv2.circle(_debug_im, (p1[1], p1[0]), radius=1, color=(0, 255, 0), thickness=2)
                cv2.circle(_debug_im, (p2[1], p2[0]), radius=1, color=(0, 255, 0), thickness=2)

                cv2.line(_debug_im, (p1[1], p1[0]), (p2[1], p1[0]), color=(0, 0, 255), thickness=2)

            cv2.imwrite(out_fn, _debug_im)

        return pair_points, self.max_traveled_pixel, self.max_pair_distance, self.keypoint_to_boundary_distance


if __name__ == '__main__':
    import PIL.Image as Image
    from evaluate import *
    from glob import glob

    closing_model = ClosingModel()
    paths = glob('D:/data/geek/close_contours/input/*')
    ca_path = 'D:/data/geek/close_contours/ca.csv',
    output = {
                 'hor01_004_k_A.A0019.png': {},
                 'hor01_018_021_k_A.A0005.png': {'1356_1862': '1364_1868', '200_10': '10_0'}
             },
    report_path = 'report.xlsx'
    for p in paths:
        sketch_tgt_im = np.array(Image.open(p).convert('L'))
        _pair_points = closing_model.process(sketch_tgt_im)
