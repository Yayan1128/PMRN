import numpy as np
import torch


def data_load(train_file, test_file):
    """
    load data from txt file
    :param train_file: path of train file
    :param test_file:  path of test file
    :return:
    """
    tr_img_path = []
    ts_img_path = []
    tr_labels = []
    ts_labels = []

    with open(train_file, 'r') as f:
        tr_data = [l.strip() for l in f.readlines()]

    for line in tr_data:
        image = line.split()
        tr_img_path.append(image[0])

    with open(test_file, 'r') as f:
        ts_data = [l.strip() for l in f.readlines()]

    for line in ts_data:
        data = line.split()
        ts_img_path.append(data[0])
        # print(int(data[1]))
        ts_labels.append(int(data[1]))


    for i in range(0, len(tr_img_path)):
        tr_labels.append(0)
    # print(ts_img_path)
    # print(ts_labels)
    # loc = np.where(np.array(ts_labels) == 0)
    # print(loc)
    # # # print(loc.shape())
    # print(loc[0][4])
    return [tr_img_path, tr_labels, ts_img_path, ts_labels]

