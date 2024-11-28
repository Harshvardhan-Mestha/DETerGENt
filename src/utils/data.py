"""
This module contains data loading utilities.
- load_data: Loads class-wise data from a csv file.
- preprocess: Preprocesses the data to a 80-20 split.
"""

import pandas as pd


CLASS_NAMES_SHORT = ["ATCS", "CLFS", "CRDM", "COPD", "LNGN", "MSTL", "PLEF", "PNUM", "PNTX", "TUBC"]


def _load_data(data_path="data/xray_data.csv"):
    """
    Loads class-wise data from a csv file.
    
    Data returned has the following structure:

    [
        [(x, y, e, case), (x, y, e, case), ...], # class 1
        [(x, y, e, case), (x, y, e, case), ...], # class 2
        ...
    ]

    There can be overlap between classes.

    Args:
        data_path (str, optional)
            Path to the data CSV. Defaults to "data/xray_data.csv".
            The CSV must have "img_dir", "label", "desc", "desc_pth", "diagnosis", "certainty", "label_short" columns.

        split (float, optional)
            Split ratio. Defaults to 0.25.

    Returns:
        D_classwise: List
            List of class-wise splits
    """

    # read csv
    data = pd.read_csv(data_path, index_col=False)

    D_ovr = [
        list(data["img_dir"]),
        list(data["label"]),
        list(data["desc"]),
        list(data["case"]),
        list(data["desc_pth"]),
        list(data["diagnosis"]),
        list(data["certainty"]),
        list(data["label_short"]),
    ]

    # (x, y, e, extra), converted because easy to use.
    class_idx = []
    class_data = list(data["label_short"])

    # iterate over the dataframe and get indices of classes
    for cname in CLASS_NAMES_SHORT:
        l = []
        for i in range(len(class_data)):
            if cname in class_data[i]:
                l.append(i)
        class_idx.append(l)
        print(f"[INFO] {cname} has {len(l)} examples.")

    # collect data in a list of (x, y, e, case) tuples
    D = []
    for i in range(len(data)):
        D.append((D_ovr[0][i], D_ovr[1][i], D_ovr[2][i], D_ovr[3][i]))

    D_classwise = []
    # make list of a class wise splits
    for idxs in class_idx:
        D_temp = []
        for i in idxs:
            D_temp.append(D[i])
        D_classwise.append(D_temp)

    print("[INFO] Data loaded.")
    return D_classwise


def load_data():
    """
    Returns the whole dataset.
    """

    D = pd.read_csv("data/xray_data.csv", index_col=False)
    D.drop(["Unnamed: 0", "desc_pth", "diagnosis", "certainty", "label_short"], axis=1, inplace=True)

    D = [
        list(D["img_dir"]),
        list(D["label"]),
        list(D["desc"]),
        list(D["case"]),
    ]
    data = []
    for i in range(len(D[0])):
        data.append((D[0][i], D[1][i], D[2][i], D[3][i]))

    print("[INFO] Data processed.")
    return data
