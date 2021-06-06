import csv
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_data(file_dir: str):
    sentences = []
    labels = []
    for f_name in os.listdir(file_dir):
        if f_name.endswith('.tsv'):
            print(f_name)
            file_path = Path(os.path.join(file_dir, f_name))
            with open(file_path, encoding='utf-16', mode='r') as in_file:
                for line in in_file:
                    vals = line.strip().split('\t')
                    sentences.append([vals[0], vals[1]])
                    labels.append(vals[2])
        X_train, X_valid, y_train, y_valid = train_test_split(sentences, labels, test_size=0.1, random_state=42)
        with open('new_train/' + f_name, encoding="utf-16", mode='w', newline='') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for i in range(len(y_train)):
                tsv_writer.writerow([X_train[i][0], X_train[i][1], y_train[i]])
        with open('new_valid/valid_' + f_name, encoding="utf-16", mode='w', newline='') as out_file2:
            tsv_writer = csv.writer(out_file2, delimiter='\t')
            for i in range(len(y_valid)):
                tsv_writer.writerow([X_valid[i][0], X_valid[i][1], y_valid[i]])
        sentences.clear()
        labels.clear()


def main():
    split_data("train_data")


main()
