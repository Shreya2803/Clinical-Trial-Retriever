import os
import pandas as pd

def read_text_files_from_directory(directory_path):
    """
    Reads all text files in a given directory and returns their content.
    """
    texts = []
    filenames = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
                filenames.append(filename)

    return texts, filenames

def load_data_from_folders(root_folder_path):
    """
    Loads text files from multiple folders and returns a DataFrame.
    """
    all_texts = []
    all_filenames = []
    all_folders = []

    for foldername, subfolders, filenames in os.walk(root_folder_path):
        print(f'Processing folder: {foldername}')
        texts, file_names = read_text_files_from_directory(foldername)
        all_texts.extend(texts)
        all_filenames.extend(file_names)
        all_folders.extend([foldername] * len(file_names))  # Keep track of folder names

    
    data_df = pd.DataFrame({
        'folder': all_folders,
        'filename': all_filenames,
        'text': all_texts
    })
    data_df['docno'] = data_df['filename']#

    return data_df

# TREC Documents Path
root_folder_path = '/content/drive/MyDrive/TREC2023/ClinicalTrials.2023-05-08.trials0'
data_df = load_data_from_folders(root_folder_path)
docs = data_df.to_dict(orient='records')#
print("Data DataFrame:")
print(data_df.head())
