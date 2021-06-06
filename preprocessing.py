import pandas as pd
import preprocessor as p



def read_labels(labels, labels_path):
    f_labels = open(labels_path, 'r')
    line = f_labels.readline()
    
    while line != '':
        parts = line.strip().split(',')
        labels[parts[0]] = parts[1]
        line = f_labels.readline()

def preprocess_file(path_to_file, output_file, enc, labels=None):
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.HASHTAG, p.OPT.ESCAPE_CHAR, p.OPT.SMILEY, p.OPT.RESERVED)
    f = open(path_to_file, 'r', encoding=enc)
    f.readline()
    line = f.readline()
    f_out = open(output_file, 'w', encoding='utf-16')
    while line != '':
        parts = line.split('\t')
        #print(parts)
        new_tweet = p.tokenize(parts[1])
        if labels is not None:
            label = labels[parts[0]]
            new_line = parts[0] + "\t" + new_tweet + "\t" + label + "\n"
        else:
            new_line = parts[0] + "\t" + new_tweet + "\t" + parts[2] + "\n"
        f_out.write(new_line)
        line = f.readline()
    f_out.close()


labels_file = None
output_file = 'english_train_set.txt'
input_file = './OLIDv1/olid-training-v1.0.tsv'

if labels_file is not None:
    labels = {}
    read_labels(labels, labels_file)
    preprocess_file(input_file, output_file, 'utf-16', labels)
else:
    preprocess_file(input_file, output_file, 'utf-8')



