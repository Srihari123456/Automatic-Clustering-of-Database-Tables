import os
import csv
import pandas as pd
import numpy as np
import pickle
import random
from sklearn.preprocessing import LabelEncoder
from tokenizer import SmartTokenizer

def process_csv_files(fileList, outputFileName, tokenize=True, dataIncluded=True, onlyStringColumns=True, p=0.5):
    tokenizer = SmartTokenizer()
    token_list = []
    fc = 0
    for filename in fileList:
        if filename.endswith(".csv"):
            print("Processing ", filename, " count = ", fc)
            fc = fc + 1
            with open(os.path.join("../csv", filename), 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                table_name = filename.split('.')[0]
                column_names = next(csv_reader)  # Read header row

                if onlyStringColumns:
                    # Read the CSV file into a pandas DataFrame
                    df = pd.read_csv(file)
                    # Get columns with string data type
                    string_columns = identify_string_columns(df)
                    # Create a new DataFrame with only string columns
                    df_string = df[string_columns]

                table_info = table_name + " " + " ".join(column_names)
                if dataIncluded:
                    # Read data rows
                    if onlyStringColumns:
                        data_rows = df_string.values.tolist()  # Convert DataFrame to list of lists
                    else:
                        data_rows = list(csv_reader)  # Read all data rows into a list
                    num_rows = len(data_rows)
                    sample_size = int(num_rows * p)  # Calculate the sample size based on the fraction 'p'
                    sampled_rows = random.sample(data_rows, k=sample_size)  # Sample rows without replacement

                    for row in sampled_rows:
                        # Combine table name, column names, and data row
                        if onlyStringColumns:
                            table_info += " " + " ".join(map(str, row))
                        else:
                            table_info += " " + " ".join(row)

                tokens = tokenizer.tokenize(table_info)

                # Remove tokens which aren't alphanumeric and convert alphanumeric tokens to lowercase
                tokens_alphanum = [t.lower() for t in tokens if t.isalnum()]
                if not tokenize:
                    token_string = " ".join(tokens_alphanum)
                    token_list.append(token_string)
                else:
                    token_list.append(tokens_alphanum)

    with open(outputFileName, "wb") as f:
        pickle.dump(token_list, f)

def identify_string_columns(df):
    # Get the data types of all columns
    column_data_types = df.dtypes

    # Initialize a list to store string columns
    string_columns = []

    # Iterate over each column's data type
    for column, data_type in column_data_types.items():
        # Check if the data type is object (string)
        if data_type == 'object':
            string_columns.append(column)

    return string_columns


def generateEncodedLabels(df):
    label_encoder = LabelEncoder()
    print(df["label"].unique())
    df['encodedLabel'] = label_encoder.fit_transform(df['label'])
    np.save("encoded_labels.npy", df['encodedLabel'])

def main():
    train_directory = "../csv/"
    process_csv_files(os.listdir(train_directory), 'string_entire_data_included_train_list.pkl', p=1)

    # test_df = pd.read_csv("test_set.csv")
    # process_csv_files(test_df['filename'].to_list(), 'test_list.pkl', False)

    # generateEncodedLabels(test_df)

if __name__ == "__main__":
    csv.field_size_limit(10000000000)
    main()
