import os
import pandas as pd
import shutil

def organize_dataset(dataset_path: str):
    """
    Organizes the dataset by creating directories for each label and copying images into these directories.

    Args:
        dataset_path (str): The root directory path of the dataset.
    """
    path_image_dir = os.path.join(dataset_path, 'image')
    path_csv_dir = os.path.join(dataset_path, 'csv')

    # get image dir path list
    list_name_image_dir = os.listdir(path_image_dir)
    list_path_image_dir = []
    for name_dir in list_name_image_dir:
        path = os.path.join(path_image_dir, name_dir)
        list_path_image_dir.append(path)

    # get image dir path dic
    dic_path_image_file = {}
    for path_image_dir in list_path_image_dir:
        list_name_image_file = os.listdir(path_image_dir)
        for name_image_file in list_name_image_file:
            path = os.path.join(path_image_dir, name_image_file)
            name = os.path.basename(path)
            dic_path_image_file[name] = path

    # get csv file path list
    list_name_csv_file = os.listdir(path_csv_dir)
    list_path_csv_file = []
    for name_file in list_name_csv_file:
        path = os.path.join(path_csv_dir, name_file)
        list_path_csv_file.append(path)

    # Read labels_correspondence.csv
    df_class_mapping = pd.read_csv(os.path.join(dataset_path, "labels_correspondence.csv"))

    # make output folder
    path_desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    list_folder_names = df_class_mapping['class2'].unique().tolist()
    dic_folder_pathes = {}

    for folder_name in list_folder_names:
        folder_path = os.path.join(path_desktop, "output",str(folder_name))
        if not os.path.exists(folder_path):
            dic_folder_pathes[folder_name] = folder_path
            os.makedirs(folder_path)

    # merge dataframe
    dataframes = [pd.read_csv(path) for path in list_path_csv_file]

    # conbine df
    df_labels = pd.concat(dataframes, axis=0, ignore_index=True)

    # organize folder structure
    dic_class = {}

    for file_name in dic_path_image_file.keys():
        row = df_labels[df_labels['file'] == file_name].iloc[0]

        mapping = df_class_mapping[
            (df_class_mapping['type'] == row['type']) &
            (df_class_mapping['name'] == row['name']) &
            (df_class_mapping['state'] == row['state'])
        ]

        if not mapping.empty:
            name_class = mapping.iloc[0]["class2"]
        else:
            name_class = None

        dic_class[file_name] = name_class


    # Copy image to the label directory
    for file_name in dic_class.keys():
        name_class = dic_class[file_name]
        path = dic_path_image_file[file_name]
        copy_path = dic_folder_pathes[name_class]

        shutil.copy(path, copy_path)

# Example usage
organize_dataset("C:\\GitHub\\BCCProject\\BCCProject\\dataset")
