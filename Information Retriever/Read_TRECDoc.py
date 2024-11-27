import pandas as pd
import os


base_path = '/content/drive/MyDrive/TREC2023/ClinicalTrials.2023-05-08.trials0/'
def read_documents_from_results(results_df):
    for _, row in results_df.iterrows():
        docno = row['docno']

        folder_name = docno[:7] + 'xxxx'
        documents_path = os.path.join(base_path, folder_name, docno)

        try:
            with open(documents_path, 'r') as file:
                # Read the first 10 lines
                lines = [file.readline().strip() for _ in range(10)]
                print(f"Document: {docno}")
                print("\n".join(lines))
                print("\n" + "-" * 50 + "\n")
        except FileNotFoundError:
            print(f"Document {docno} not found.")

read_documents_from_results(initial_results.head(100))
