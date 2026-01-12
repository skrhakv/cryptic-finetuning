import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import csv
from torch import nn
from sklearn.utils import class_weight
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import sys 

sys.path.append('/home/skrhakv/cryptic-nn/src')
import baseline_utils
import finetuning_utils

torch.manual_seed(0)

DATASET = 'cryptobench'
DATA_PATH = f'/home/skrhakv/cryptic-nn/data/{DATASET}'
ESM_EMBEDDINGS_PATH = f'{DATA_PATH}/embeddings'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from transformers import EsmModel, AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
import csv
from datasets import Dataset
import functools
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
import gc
import bitsandbytes as bnb
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings('ignore')
torch.manual_seed(42)
MODEL_NAME = 'facebook/esm2_t36_3B_UR50D'


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
finetuned_model = finetuning_utils.FinetunedEsmModelWithCnn(MODEL_NAME, in_channels=finetuning_utils.NUMBER_OF_MODES).half().to(device)

nma_path = '/home/skrhakv/cryptic-nn/data/cryptobench/fluctuation/fluctuation'
nma_scaler = finetuning_utils.train_scaler('/home/skrhakv/cryptic-nn/data/cryptobench/fluctuation/train.txt', nma_path=nma_path)
train_dataset = finetuning_utils.process_sequence_dataset('/home/skrhakv/cryptic-nn/data/cryptobench/fluctuation/train.txt', tokenizer, nma_path=nma_path, nma_scaler=nma_scaler)
test_dataset = finetuning_utils.process_sequence_dataset('/home/skrhakv/cryptic-nn/data/cryptobench/fluctuation/test.txt', tokenizer, nma_path=nma_path, nma_scaler=nma_scaler)
# TODO: edit collate fn to include nma
partial_collate_fn = functools.partial(finetuning_utils.collate_fn, tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=partial_collate_fn)
val_dataloader = DataLoader(test_dataset, batch_size=test_dataset.num_rows, collate_fn=partial_collate_fn)

optimizer = bnb.optim.AdamW8bit(finetuned_model.parameters(), lr=0.0001, eps=1e-4) 

EPOCHS = 3

# precomputed class weights
class_weights = torch.tensor([0.5303, 8.7481], device=device)
cbs_loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
plDDT_loss_fn = nn.MSELoss() 
distance_loss_fn = nn.MSELoss()

for name, param in finetuned_model.named_parameters():
     if name.startswith('llm'): 
        param.requires_grad = False

test_losses = []
train_losses = []

for epoch in range(EPOCHS):
    if epoch > 1:
        for name, param in finetuned_model.named_parameters():
            param.requires_grad = True

    finetuned_model.eval()

    # VALIDATION LOOP
    with torch.no_grad():
        for batch in val_dataloader:
            output1 = finetuned_model(batch)

            labels = batch['labels'].to(device)
    
            flattened_labels = labels.flatten()

            cbs_logits = output1.flatten()[flattened_labels != -100]
            valid_flattened_labels = labels.flatten()[flattened_labels != -100]

            predictions = torch.round(torch.sigmoid(cbs_logits)) # (probabilities>0.95).float() # torch.round(torch.sigmoid(valid_flattened_cbs_logits))

            cbs_test_loss =  cbs_loss_fn(cbs_logits, valid_flattened_labels)

            test_loss = cbs_test_loss

            test_losses.append(test_loss.cpu().float().detach().numpy())

            # compute metrics on test dataset
            test_acc = baseline_utils.accuracy_fn(y_true=valid_flattened_labels,
                                   y_pred=predictions)
            fpr, tpr, thresholds = metrics.roc_curve(valid_flattened_labels.cpu().float().numpy(), torch.sigmoid(cbs_logits).cpu().float().numpy())
            roc_auc = metrics.auc(fpr, tpr)

            mcc = metrics.matthews_corrcoef(valid_flattened_labels.cpu().float().numpy(), predictions.cpu().float().numpy())

            f1 = metrics.f1_score(valid_flattened_labels.cpu().float().numpy(), predictions.cpu().float().numpy(), average='weighted')

            precision, recall, thresholds = metrics.precision_recall_curve(valid_flattened_labels.cpu().float().numpy(), torch.sigmoid(cbs_logits).cpu().float().numpy())
            auprc = metrics.auc(recall, precision)

            del labels, cbs_logits, valid_flattened_labels, flattened_labels
            gc.collect()
            torch.cuda.empty_cache()
    
    finetuned_model.train()

    batch_losses = []

    # TRAIN

    for batch in train_dataloader:

        output1 = finetuned_model(batch)

        labels = batch['labels'].to(device)

        flattened_labels = labels.flatten()

        cbs_logits = output1.flatten()[flattened_labels != -100]
        valid_flattened_labels = labels.flatten()[flattened_labels != -100]

        cbs_loss =  cbs_loss_fn(cbs_logits, valid_flattened_labels)

        loss = cbs_loss
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_losses.append(loss.cpu().float().detach().numpy())
        
        del labels, output1, cbs_logits, valid_flattened_labels, flattened_labels
        gc.collect()
        torch.cuda.empty_cache()

    train_losses.append(sum(batch_losses) / len(batch_losses))
    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {test_acc:.2f}% | Test loss: {test_loss:.5f}, AUC: {roc_auc:.4f}, MCC: {mcc:.4f}, F1: {f1:.4f}, AUPRC: {auprc:.4f}, sum: {sum(predictions.to(dtype=torch.int))}")

OUTPUT_PATH = '/home/skrhakv/cryptic-nn/src/trained-models/NMA-finetuned-model-with-CNN.pt'
torch.save(finetuned_model, OUTPUT_PATH)



with torch.no_grad():
    for batch in val_dataloader:
        output1 = finetuned_model(batch)

        labels = batch['labels'].to(device)

        flattened_labels = labels.flatten()

        cbs_logits = output1.flatten()[flattened_labels != -100]
        valid_flattened_labels = labels.flatten()[flattened_labels != -100]

        predictions = torch.round(torch.sigmoid(cbs_logits)) # (probabilities>0.95).float() # torch.round(torch.sigmoid(valid_flattened_cbs_logits))

        cbs_test_loss =  cbs_loss_fn(cbs_logits, valid_flattened_labels)

        test_loss = cbs_test_loss

        test_losses.append(test_loss.cpu().float().detach().numpy())

        # compute metrics on test dataset
        test_acc = baseline_utils.accuracy_fn(y_true=valid_flattened_labels,
                                y_pred=predictions)
        fpr, tpr, thresholds = metrics.roc_curve(valid_flattened_labels.cpu().float().numpy(), torch.sigmoid(cbs_logits).cpu().float().numpy())
        roc_auc = metrics.auc(fpr, tpr)

        mcc = metrics.matthews_corrcoef(valid_flattened_labels.cpu().float().numpy(), predictions.cpu().float().numpy())

        f1 = metrics.f1_score(valid_flattened_labels.cpu().float().numpy(), predictions.cpu().float().numpy(), average='weighted')

        precision, recall, thresholds = metrics.precision_recall_curve(valid_flattened_labels.cpu().float().numpy(), torch.sigmoid(cbs_logits).cpu().float().numpy())
        auprc = metrics.auc(recall, precision)

print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {test_acc:.2f}% | Test loss: {test_loss:.5f}, AUC: {roc_auc:.4f}, MCC: {mcc:.4f}, F1: {f1:.4f}, AUPRC: {auprc:.4f}, sum: {sum(predictions.to(dtype=torch.int))}")
