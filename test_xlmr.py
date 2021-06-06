import csv
import os
import pandas as pd
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any, Callable
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig, XLMRobertaForSequenceClassification
from pathlib import Path


class SpamDataset(Dataset):
    def __init__(self, sentences: List[str], labels: List[int], tokenizer: XLMRobertaTokenizer, max_len: int = 128):
        if len(sentences) != len(labels):
            raise RuntimeError("Sentences and labels should have the same number of elements.")
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.labels = labels
        self.max_len = max_len

    def __getitem__(self, index: int):
        inputs = self.tokenizer(self.sentences[index], truncation=True, pad_to_max_length=True, max_length=self.max_len)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long).squeeze(),
            "attention_mask": torch.tensor(mask, dtype=torch.long).squeeze(),
            "labels": torch.tensor(self.labels[index], dtype=torch.long)
        }

    def __len__(self):
        return len(self.sentences)

    @staticmethod
    def load_from_csv(file_dir: str, tokenizer: XLMRobertaTokenizer, max_len: int) -> "SpamDataset":
        sentences = []
        labels = []
        label_to_index = {}
        for f_name in os.listdir(file_dir):
            if f_name.endswith('.tsv'):
                print(f_name)
                file_path = Path(os.path.join(file_dir, f_name))
                with file_path.open(encoding="ISO-8859-1") as file:
                    reader = csv.reader(file, delimiter="\t", strict=True)
                    for idx, row in enumerate(reader):
                        sentences.append(row[1])
                        if row[2] not in label_to_index:
                            if not label_to_index:
                                label_to_index[row[2]] = 0
                            else:
                                label_to_index[row[2]] = len(label_to_index)
                        labels.append(label_to_index[row[2]])
        return SpamDataset(sentences, labels, tokenizer, max_len)

    @staticmethod
    def pd_load_tsv(file_dir: str, tokenizer: XLMRobertaTokenizer, max_len: int) -> "SpamDataset":
        sentences = []
        labels = []
        label_to_index = {'NOT': 0,
                          'OFF': 1}
        for f_name in os.listdir(file_dir):
            if f_name.endswith('.tsv'):
                print(f_name)
                file_path = Path(os.path.join(file_dir, f_name))
                df = pd.read_csv(file_path, sep='\t', header=None)
                print(df[1].head(5))
                sentences.extend(list(df[1]))
                labs = [label_to_index[i] for i in list(df[2])]
                #df = df.assign(Label=lambda x: label_to_index[x[2]])
                labels.extend(labs)
        return SpamDataset(sentences, labels, tokenizer, max_len)

    @staticmethod
    def load_manually(file_dir: str, tokenizer: XLMRobertaTokenizer, max_len: int) -> "SpamDataset":
        sentences = []
        labels = []
        label_to_index = {'NOT': 0,
                          'OFF': 1}
        for f_name in os.listdir(file_dir):
            if f_name.endswith('.tsv'):
                print(f_name)
                file_path = Path(os.path.join(file_dir, f_name))
                with open(file_path, encoding='utf-16', mode='r') as in_file:
                    for line in in_file:
                        vals = line.strip().split('\t')
                        sentences.append(vals[1])
                        labels.append(label_to_index[vals[2]])
        return SpamDataset(sentences, labels, tokenizer, max_len)


def test_model(model: nn.Module, device: torch.device, data_loader: DataLoader) -> Tuple[List[int], List[int]]:
    model.eval()

    labels_list = []
    preds_list = []

    with torch.no_grad():
        for batch in data_loader:
            ids, mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(
                device)
            ret_dict = model.forward(ids, attention_mask=mask, labels=labels, return_dict=True)
            _, preds = torch.max(ret_dict['logits'], dim=1)

            labels_list.extend(labels.cpu().numpy().tolist())
            preds_list.extend(preds.cpu().numpy().tolist())

    return labels_list, preds_list


def valid_epoch(model: nn.Module, device: torch.device, data_loader: DataLoader) -> Dict[str, float]:
    return compute_metrics(*test_model(model, device, data_loader))


def compute_metrics(labels: List[int], predictions: List[int]) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    conf_matrix = get_confusion_matrix(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix
    }


def get_confusion_matrix(labels: List[int], predictions: List[int]):
    return confusion_matrix(labels, predictions)


def create_model(scope: Dict[str, Any]) -> Tuple[XLMRobertaForSequenceClassification, XLMRobertaTokenizer]:
    device = scope["devices"][0]
    model_name = "xlm-roberta-base"
    config = XLMRobertaConfig.from_pretrained(model_name)
    config.num_labels = 2  # spam, ham

    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaForSequenceClassification.from_pretrained(model_name, config=config)
    model_checkpoint = torch.load(scope['model_path'])
    model.load_state_dict(model_checkpoint)
    model.to(device)
    model.eval()
    return model, tokenizer


def main(scope):
    device = scope["devices"][0]
    model, tokenizer = create_model(scope)

    test_dataset = SpamDataset.load_manually(scope['test_dir'], tokenizer, scope['max_len'])
    test_loader = DataLoader(test_dataset, batch_size=scope['batch_size'], shuffle=False, num_workers=2)
    print(valid_epoch(model, device, test_loader))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--test_dir")
    parser.add_argument("--gpu")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    scope = vars(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu is not None and len(args.gpu) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((str(x) for x in args.gpu))
        # scope["devices"] = [torch.device("cuda", int(x)) for x in args.gpu]
        scope["devices"] = [torch.device("cuda", x) for x in range(len(args.gpu))]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        scope["devices"] = [torch.device("cpu")]
    print("Starting on:")
    for x in scope["devices"]:
        print(f"\t{x}")
    print()
    main(scope)