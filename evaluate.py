import csv


def evaluate(ca_path, output, report_path=None, **kwargs):
    """Evaluates closing contours.

    Args:
        ca_path (str): file path to ca (.csv)
        output (dict): in the form {'img_name_with_extension': {'row1_col1': 'row2_col2', 'row1_col1': 'row2_col2', ...}
        report_path (str): file path to report (.xlsx). If report_path is not None, write report to file.

    Returns:
        img_report (dict): the result of each image
        row_report (dict): the result of each point
        summary_report (dict): the overall result

    """
    # Read from CA.
    with open(ca_path) as csv_file:
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

    img_report = {}
    row_report = {}
    total_img = len(output)
    total_correct_img = total_points = total_fn = total_fp = total_tp = 0
    for img_name, output_coords in output.items():
        # For each new image, set True Positive to 0.
        tp = 0
        img_report[img_name] = {
            'precision': '-',
            'recall': '-',
            'n_contours': 0,
            'n_output': 0,
            'correct': 0,
            'overall': 0
        }

        # Initally, set False Negative equal to number of contours in CA and decrement by one as the process goes on.
        img_report[img_name]['n_contours'] = fn = ca_dict[img_name]['n_contours']
        # Similarly, set False Positive equal to number of output pairs.
        img_report[img_name]['n_output'] = fp = len(output_coords)
        # Increment the number of total segments.
        total_points += fn
        # Get the inverted dictionary.
        inverted_output = {y: x for x, y in output_coords.items()}

        # Process each segment in each image in CA.
        for info in ca_dict[img_name]['coor_info']:
            coor1, coor2, row_index = info
            result = 0  # if the pair is correct, result = 1. Otherwise, 0.
            if output_coords.get(coor1) == coor2 or inverted_output.get(coor1) == coor2:
                tp += 1
                fn -= 1
                fp -= 1
                result = 1
            row_report[row_index] = result

        # Only calculate P and R if n_contours is not 0.
        if ca_dict[img_name]['n_contours'] != 0:
            precision = tp * 1.0 / (tp + fp)
            recall = tp * 1.0 / (tp + fn)
            img_report[img_name]['precision'] = '%.2f' % precision
            img_report[img_name]['recall'] = '%.2f' % recall

        img_report[img_name]['correct'] = tp
        if img_report[img_name]['correct'] == img_report[img_name]['n_output']:
            img_report[img_name]['overall'] = 1
            total_correct_img += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn

    total_precision = total_recall = percent_correct = 0.00
    if total_points != 0:
        total_precision = total_tp * 1.0 / (total_tp + total_fp)
        total_recall = total_tp * 1.0 / (total_tp + total_fn)
        percent_correct = total_tp * 1.0 / total_points
    percent_correct_img = total_correct_img * 1.0 / total_img
    _summary_report = {
        'overall_precision_point': '%.2f' % total_precision,
        'overall_recall_point': '%.2f' % total_recall,
        'percent_correct_point': '%.2f' % percent_correct,
        '-': '-',
        'total_img': total_img,
        'total_correct_img': total_correct_img,
        'percent_correct_img': '%.2f' % percent_correct_img
    }
    summary_report = {**kwargs, **_summary_report}

    if report_path:
        from openpyxl import Workbook
        wb = Workbook()
        ws1 = wb.active
        ws1.title = 'point_report'
        titles = ['File_Name', 'Cut_Name', 'Cut_Reference', 'Image_reference', 'Number_of_Open_Contours',
                  'Start_Row_(Y)', 'Start_Col_(X)', 'End_Row_(Y)', 'End_Col_(X)', 'Result']
        ws1.append(titles)
        with open(ca_path) as csv_file:
            ca = list(csv.reader(csv_file, delimiter=','))
        csv_file.close()
        for i, ca_row in enumerate(ca[1:]):
            new_row = ca_row + [row_report.get(i + 1)]
            ws1.append(new_row)

        ws2 = wb.create_sheet('img_report')
        ws2.append(('Image_Name', 'n_contours', 'n_output', 'Total_Correct', 'Precision', 'Recall', 'Overall'))
        for img_name, report_info in img_report.items():
            ws2.append(
                (
                    img_name,
                    report_info['n_contours'],
                    report_info['n_output'],
                    report_info['correct'],
                    report_info['precision'],
                    report_info['recall'],
                    report_info['overall']
                )
            )

        ws3 = wb.create_sheet('summary')
        for k, v in summary_report.items():
            ws3.append((k, v))
        wb.save(report_path)

    return img_report, row_report, summary_report


if __name__ == '__main__':
    import os
    import numpy as np
    from PIL import Image
    from glob import glob
    from datetime import datetime
    from algo.rule1.closing import ClosingModel
    from algo.rule1.utils import convert_output_format

    closing_model = ClosingModel()
    # TODO
    paths = glob('D:/data/geek/close_contours/input/*')
    # TODO
    ca_path = 'D:/data/geek/close_contours/ca.csv'

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    # TODO
    report_path = 'D:/output/geek/close_contours/report_%s.xlsx' % dt_string
    result = {}
    for i, p in enumerate(paths[29:]):
        print(i, p)
        sketch_tgt_im = np.array(Image.open(p).convert('L'))
        pair_points, max_traveled_pixels, max_pair_distance, keypoint_to_boundary_distance = \
            closing_model.process(sketch_tgt_im)
        name = os.path.basename(p)
        result[name] = convert_output_format(pair_points)
    kwargs = {
        'max_traveled_pixels': max_traveled_pixels,
        'max_pair_distance': max_pair_distance,
        'keypoint_to_boundary_distance': keypoint_to_boundary_distance,
        '-': '-'
    }
    evaluate(ca_path, result, report_path, **kwargs)


    # ca_path = 'D:/data/geek/close_contours/ca.csv',
    # output = {
    #  'hor01_004_k_A.A0019.png': {},
    #  'hor01_018_021_k_A.A0005.png': {'1356_1862': '1364_1868', '200_10': '10_0'}
    # },
    # report_path = 'report.xlsx'

