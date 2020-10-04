def get_intersection(bb1, bb2):
    """

    :param bb1:
    :param bb2:
    :return:
    """
    x_left = max(bb1[0][0], bb2[0][0])
    x_right = min(bb1[1][0], bb2[1][0])

    y_top = max(bb1[0][1], bb2[0][1])
    y_bottom = min(bb1[1][1], bb2[1][1])

    z_up = max(bb1[0][2], bb2[0][2])
    z_down = min(bb1[1][2], bb2[1][2])

    if x_right < x_left or y_bottom < y_top or z_down < z_up:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top) * (z_down -
                                                                   z_up)

    return intersection_area


def get_iou(bb1, bb2, area1, area2):
    """

    :param bb1:
    :param bb2:
    :param area1:
    :param area2:
    :return:
    """
    intersection_area = get_intersection(bb1, bb2)
    return intersection_area / float(area1 + area2 - intersection_area)


def get_bounding_box_volume(bb):
    """

    :param bb:
    :return:
    """
    width = bb[1][0] - bb[0][0]
    depth = bb[1][1] - bb[0][1]
    height = bb[1][2] - bb[0][2]
    return width * depth * height
