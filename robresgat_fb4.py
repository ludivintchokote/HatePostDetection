#implementation on fb without GAT
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
import math
import pandas as pd
import os
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torch
import random
import functools
import operator
import cv2
from advt.attack import DeepFool
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import collections
import torchvision.models as models
import torch.nn.functional as F
# from advertorch.attacks import carlini_wagner
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, BertTokenizer, BertModel, VisualBertModel, VisualBertConfig, RobertaTokenizer, RobertaModel
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm, trange
from dataloader_adv_train import meme_dataset
import json
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

# TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
TOKENIZER = RobertaTokenizer.from_pretrained('roberta-base')
DROPOUT = 0.2
HIDDEN_SIZE = 128 #128
BATCH_SIZE = 8 #7 #16 #8
NUM_LABELS = 2
NUM_EPOCHS = 15 #30 20 #50
EARLY_STOPPING = {"patience": 30, "delta": 0.005}
LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1]
WEIGHT_DECAY = 0.01
WARMUP = 0.06
INPUT_LEN = 768
VIS_OUT = 2048
# VIS_OUT = 1280
# criterion = nn.CrossEntropyLoss().cuda()

EXP_NAME = 'roberta_resnet_base'


class IncrementalSMOTEMemeDataset(Dataset):
    def __init__(self, original_dataset, n_components=512, batch_size=200, save_path='synthetic_samples.npz'):
        self.original_dataset = original_dataset
        self.tokenizer = original_dataset.tokenizer
        self.n_components = n_components
        self.batch_size = batch_size
        # self.save_path = save_path

        # Check if synthetic samples exist
        # if os.path.exists(self.save_path):
        #     print("Loading synthetic samples from disk...")
        #     data = np.load(self.save_path)
        #     self.features_resampled = data['features']
        #     self.labels_resampled = data['labels']
        # else:
        self.labels = []
        self.original_samples = []
        self.features_resampled = None
        self.labels_resampled = None

        print("Preparing dataset for SMOTE...")

        # Initialize Incremental PCA and StandardScaler
        self.image_pca = IncrementalPCA(n_components=n_components // 2)
        self.text_pca = IncrementalPCA(n_components=n_components // 2)
        self.image_scaler = StandardScaler()
        self.text_scaler = StandardScaler()

        # Process data in batches
        for i in tqdm(range(0, len(original_dataset), batch_size), desc="Processing batches"):
            batch_image_features = []
            batch_text_features = []
            batch_end = min(i + batch_size, len(original_dataset))

            for j in range(i, batch_end):
                sample = original_dataset[j]

                img_feat = sample['image'].flatten().cpu().numpy().astype(np.float32)
                text_feat = sample['text']['input_ids'].flatten().cpu().numpy().astype(np.float32)

                batch_image_features.append(img_feat)
                batch_text_features.append(text_feat)

                self.labels.append(sample['slabel'])
                self.original_samples.append(sample)

            # Partial fit for scaling and PCA
            self.image_scaler.partial_fit(batch_image_features)
            self.text_scaler.partial_fit(batch_text_features)

            batch_image_features_scaled = self.image_scaler.transform(batch_image_features)
            batch_text_features_scaled = self.text_scaler.transform(batch_text_features)

            self.image_pca.partial_fit(batch_image_features_scaled)
            self.text_pca.partial_fit(batch_text_features_scaled)

        # Transform all data
        all_features = []
        for i in tqdm(range(0, len(original_dataset), batch_size), desc="Transforming data"):
            batch_image_features = []
            batch_text_features = []
            batch_end = min(i + batch_size, len(original_dataset))

            for j in range(i, batch_end):
                sample = original_dataset[j]

                img_feat = sample['image'].flatten().cpu().numpy().astype(np.float32)
                text_feat = sample['text']['input_ids'].flatten().cpu().numpy().astype(np.float32)

                batch_image_features.append(img_feat)
                batch_text_features.append(text_feat)

            batch_image_features_scaled = self.image_scaler.transform(batch_image_features)
            batch_text_features_scaled = self.text_scaler.transform(batch_text_features)

            batch_image_reduced = self.image_pca.transform(batch_image_features_scaled)
            batch_text_reduced = self.text_pca.transform(batch_text_features_scaled)

            batch_features = np.hstack([batch_image_reduced, batch_text_reduced])
            all_features.extend(batch_features)

        all_features = np.array(all_features)
        self.labels = np.array(self.labels)

        print("Original class distribution:", Counter(self.labels))

        # Apply SMOTE on reduced features
        print("Applying SMOTE...")
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        self.features_resampled, self.labels_resampled = smote.fit_resample(all_features, self.labels)

        print("Resampled class distribution:", Counter(self.labels_resampled))

        # Store indices of original samples
        self.original_indices = np.where(np.arange(len(self.labels_resampled)) < len(self.labels))[0]

        # Store shapes for reconstruction
        self.original_image_shape = self.original_samples[0]['image'].shape
        self.original_text_shape = self.original_samples[0]['text']['input_ids'].shape

        # Store reference shapes from first sample
        first_sample = original_dataset[0]
        self.original_image_shape = first_sample['image'].shape
        self.original_text_shape = first_sample['text']['input_ids'].shape

        # Verify all samples have consistent shapes
        for i in range(len(original_dataset)):
            sample = original_dataset[i]
            assert sample['image'].shape == self.original_image_shape, f"Inconsistent image shape at index {i}"
            assert sample['text']['input_ids'].shape == self.original_text_shape, f"Inconsistent text shape at index {i}"
            # Save synthetic samples


    def __len__(self):
        return len(self.labels_resampled)

    def __getitem__(self, idx):
        if idx in self.original_indices:
            # For original samples
            sample = self.original_samples[idx]
            return sample
        else:
            # For synthetic samples
            feature = self.features_resampled[idx]
            label = self.labels_resampled[idx]

            # Create tensors with consistent shapes
            image = torch.zeros(self.original_image_shape, dtype=torch.float32)
            max_length = self.original_text_shape[0]

            text = {
                'input_ids': torch.zeros((max_length,), dtype=torch.long),
                'attention_mask': torch.ones((max_length,), dtype=torch.long)
            }

            return {
                'image': image,
                'text': text,
                'slabel': label,
                'feature': feature,
                'img_info': f'synthetic_{idx}'
            }

    # def save_synthetic_samples(self):
    #     print(f"Saving synthetic samples to {self.save_path}...")
    #     np.savez(self.save_path, features=self.features_resampled, labels=self.labels_resampled)


class CNN_roberta_Classifier(nn.Module):
    def __init__(self, vis_out, input_len, dropout, hidden_size, num_labels):
        super(CNN_roberta_Classifier, self).__init__()
        self.lm = RobertaModel.from_pretrained('roberta-base')
        #         self.lm = BertModel.from_pretrained('bert-base-uncased')
        self.vm = models.resnet50(pretrained=True)
        self.vm.fc = nn.Sequential(nn.Linear(vis_out, input_len))
        #         self.vm = models.efficientnet_b5(pretrained=True)
        #         self.vmlp = nn.Linear(vis_out,input_len)
        #         print(self.vm)

        embed_dim = input_len
        self.merge = torch.nn.Sequential(torch.nn.ReLU(),
                                         torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
                                         torch.nn.ReLU(),
                                         # nn.Dropout(dropout),  # Add dropout her
                                         torch.nn.Linear(2 * embed_dim, embed_dim))
        self.mlp = nn.Sequential(nn.Linear(input_len, hidden_size),
                                 nn.ReLU(),
                                 # nn.Dropout(dropout),  # Add dropout her
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, num_labels))
        self.image_space = nn.Sequential(nn.Linear(input_len, input_len),
                                         nn.ReLU(),
                                         nn.Linear(input_len, input_len),
                                         nn.ReLU(),
                                         nn.Linear(input_len, input_len))
        self.text_space = nn.Sequential(nn.Linear(input_len, input_len),
                                        nn.ReLU(),
                                        nn.Linear(input_len, input_len),
                                        nn.ReLU(),
                                        nn.Linear(input_len, input_len))

    def forward(self, image, text, label):
        #         img_cls, image_prev = self.vm(image)
        #         image = self.vmlp(image_prev)
        image = self.vm(image)
        text = self.lm(**text).last_hidden_state[:, 0, :]
        image_shifted = image
        text_shifted = text
        img_txt = (image, text)
        img_txt = torch.cat(img_txt, dim=1)
        merged = self.merge(img_txt)
        label_output = self.mlp(merged)
        return label_output, merged, image_shifted, text_shifted


def validation(dl, model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    img_names = []
    with torch.no_grad():
        for data in dl:
            data['image'] = data['image'].cuda()
            for key in data['text'].keys():
                data['text'][key] = data['text'][key].cuda()
            data['slabel'] = data['slabel'].cuda()

            predictions, _, _, _ = model(data['image'], data['text'], data['slabel'])
            outputs = predictions.argmax(1).cpu().numpy()
            targets = data['slabel'].cpu().numpy()

            fin_targets.extend(targets)
            fin_outputs.extend(outputs)
            img_names.extend(data['img_info'])

    return fin_targets, fin_outputs, img_names


def custom_collate_fn(batch):
    # Get max length for padding
    max_length = max(item['text']['input_ids'].size(-1) for item in batch)

    # Initialize lists to store batch items
    images = []
    input_ids = []
    attention_masks = []
    labels = []
    img_info = []

    for item in batch:
        # Handle image
        images.append(item['image'])

        # Handle text inputs
        curr_input_ids = item['text']['input_ids']
        curr_attention_mask = item['text']['attention_mask']

        # Ensure input_ids and attention_mask are 1D
        if len(curr_input_ids.shape) == 1:
            curr_input_ids = curr_input_ids.unsqueeze(0)
        if len(curr_attention_mask.shape) == 1:
            curr_attention_mask = curr_attention_mask.unsqueeze(0)

        # Pad sequences to max_length
        if curr_input_ids.size(1) < max_length:
            padding_length = max_length - curr_input_ids.size(1)
            curr_input_ids = F.pad(curr_input_ids, (0, padding_length), value=0)
            curr_attention_mask = F.pad(curr_attention_mask, (0, padding_length), value=0)

        input_ids.append(curr_input_ids)
        attention_masks.append(curr_attention_mask)

        # Handle labels and image info
        labels.append(item['slabel'])
        img_info.append(item['img_info'])


    # Stack all tensors
    images = torch.stack(images)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)


    return {
        'image': images,
        'text': {
            'input_ids': input_ids,
            'attention_mask': attention_masks
        },
        'slabel': labels,
        'img_info': img_info,
    }


def compute_metrics(targets, outputs):
    accuracy = accuracy_score(targets, outputs)
    f1_macro = f1_score(targets, outputs, average='macro')
    f1_micro = f1_score(targets, outputs, average='micro')

    class_report = classification_report(targets, outputs, output_dict=True)
    conf_matrix = confusion_matrix(targets, outputs)

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'class_report': class_report,
        'confusion_matrix': conf_matrix
    }


def calculate_class_weights(train_dataloader):
    all_labels = []
    for batch in train_dataloader:
        all_labels.extend(batch['slabel'].numpy())

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
    return torch.tensor(class_weights, dtype=torch.float).cuda()


def train(train_dataloader, dev_dataloader, model, optimizer, scheduler, dataset_name):
    class_weights = calculate_class_weights(train_dataloader)
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()

    max_f1_macro = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_values = []

        with tqdm(train_dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                data['image'] = data['image'].cuda()
                for key in data['text'].keys():
                    data['text'][key] = data['text'][key].cuda()
                data['slabel'] = data['slabel'].cuda()

                output, _, _, _ = model(data['image'], data['text'], data['slabel'])
                loss = criterion(output, data['slabel'])
                train_loss_values.append(loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                tepoch.set_postfix(loss=loss.item())

        avg_train_loss = sum(train_loss_values) / len(train_loss_values)
        print(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")

        # Validation step
        model.eval()
        targets, outputs, img_names = validation(dev_dataloader, model)
        metrics = compute_metrics(targets, outputs)

        print(f"Validation Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"F1 Micro: {metrics['f1_micro']:.4f}")
        print("\nPer-class F1 Scores:")
        for class_name, class_metrics in metrics['class_report'].items():
            if class_name.isdigit():
                print(f"Class {class_name}: F1-score: {class_metrics['f1-score']:.4f}")

        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])

        # Save the best model based on F1 Macro score
        if metrics['f1_macro'] > max_f1_macro:
            max_f1_macro = metrics['f1_macro']
            print(f"New best F1 Macro score: {max_f1_macro:.4f}")
            path = f'saved/{dataset_name}_weighted_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'f1_macro': max_f1_macro,
            }, path)

        print(f"Best F1 Macro score so far: {max_f1_macro:.4f}")
        print("-" * 50)


def write_test_results(outputs, image_names):
    dict_single = {}
    for i in range(len(image_names)):
        image_name = image_names[i]
        pred = str(int(outputs[i][0]))
        dict_single[image_name] = pred
    dict_single = collections.OrderedDict(sorted(dict_single.items()))
    json_object = json.dumps(dict_single, indent=4)
    json_file_name = 'preds/' + EXP_NAME + '.json'
    with open(json_file_name, "w") as outfile:
        outfile.write(json_object)


def get_torch_dataloaders(dataset_name, global_path):
    train_dataset = meme_dataset(dataset_name, 'train', TOKENIZER, None, None)
    dev_dataset = meme_dataset(dataset_name, 'val', TOKENIZER, None, None)
    test_dataset = meme_dataset(dataset_name, 'test', TOKENIZER, None, None)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    return train_dataloader, dev_dataloader, test_dataloader


def main():
    global_path = '../datasets'
    datasets = ['fb']
    #     datasets = ['mami']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset_name in datasets:

        # Get original dataloaders
        train_dataset = meme_dataset(dataset_name, 'train', TOKENIZER, None, None)
        dev_dataset = meme_dataset(dataset_name, 'val', TOKENIZER, None, None)
        test_dataset = meme_dataset(dataset_name, 'test', TOKENIZER, None, None)

        # Apply incremental SMOTE to training dataset with reduced dimensionality
        print("Applying incremental SMOTE to training dataset...")
        smote_train_dataset = IncrementalSMOTEMemeDataset(train_dataset, n_components=100, batch_size=100)

        # Apply incremental SMOTE to val dataset with reduced dimensionality
        # print("Applying incremental SMOTE to validation dataset...")
        # smote_dev_dataset = IncrementalSMOTEMemeDataset(dev_dataset, n_components=100, batch_size=100)

        # Create dataloaders
        train_dataloader = DataLoader(smote_train_dataset, batch_size=BATCH_SIZE,
                                      shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
        dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                     shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

        print(" Start Training on, ", dataset_name)
        print(f"Training samples after SMOTE: {len(train_dataloader.dataset)}")
        print(f"Validation samples: {len(dev_dataloader.dataset)}")
        print(f"Test samples: {len(test_dataloader.dataset)}")

        # Load the state dictionary from the file
        state_dict = torch.load('newly_initialized_weights.pt')

        model = CNN_roberta_Classifier(VIS_OUT, INPUT_LEN, DROPOUT, HIDDEN_SIZE, NUM_LABELS).cuda()
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-5, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.06 * (
                    len(train_dataloader) * BATCH_SIZE * NUM_EPOCHS), num_training_steps=(1 - 0.06) * (
                    len(train_dataloader) * BATCH_SIZE * NUM_EPOCHS))
        print(" Start Training on, ", dataset_name, len(train_dataloader), len(dev_dataloader), len(test_dataloader))

        checkpoint = torch.load('saved/' + 'fb' + '_weighted_best56.02' + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'], state_dict)

        class_weights = calculate_class_weights(train_dataloader)
        criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()

        # max_f1_macro = 0
        max_f1_macro = checkpoint['f1_macro']
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss_values = []

            with tqdm(train_dataloader, unit="batch") as tepoch:
                for data in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    data['image'] = data['image'].cuda()
                    for key in data['text'].keys():
                        data['text'][key] = data['text'][key].cuda()
                    data['slabel'] = data['slabel'].long().cuda()

                    output, _, _, _ = model(data['image'], data['text'], data['slabel'])
                    output = output.float()
                    loss = criterion(output, data['slabel'])
                    train_loss_values.append(loss.item())

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    tepoch.set_postfix(loss=loss.item())

            avg_train_loss = sum(train_loss_values) / len(train_loss_values)
            print(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")

            # Validation step
            model.eval()
            train_targets, train_outputs, _ = validation(train_dataloader, model)
            train_metrics = compute_metrics(train_targets, train_outputs)
            print(f"Training Metrics:")
            print(f"Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"F1 Macro: {train_metrics['f1_macro']:.4f}")
            print(f"F1 Micro: {train_metrics['f1_micro']:.4f}")
            print("\nPer-class F1 Scores (Training):")
            for class_name, class_metrics in train_metrics['class_report'].items():
                if class_name.isdigit():
                    print(f"Class {class_name}: F1-score: {class_metrics['f1-score']:.4f}")
                    print(f"  Precision: {class_metrics['precision']:.4f}")
                    print(f"  Recall: {class_metrics['recall']:.4f}")

            print("\nConfusion Matrix (Training):")
            print(train_metrics['confusion_matrix'])

            targets, outputs, img_names = validation(dev_dataloader, model)
            metrics = compute_metrics(targets, outputs)

            print(f"Validation Metrics:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Macro: {metrics['f1_macro']:.4f}")
            print(f"F1 Micro: {metrics['f1_micro']:.4f}")
            print("\nPer-class F1 Scores:")
            for class_name, metrics_dict in metrics['class_report'].items():
                if class_name.isdigit():
                    print(f"Class {class_name}: F1-score: {metrics_dict['f1-score']:.4f}")
                    print(f"  Precision: {metrics_dict['precision']:.4f}")
                    print(f"  Recall: {metrics_dict['recall']:.4f}")

            print("\nConfusion Matrix:")
            print(metrics['confusion_matrix'])

            # Save the best model based on F1 Macro score
            if metrics['f1_macro'] > max_f1_macro:
                max_f1_macro = metrics['f1_macro']
                print(f"New best F1 Macro score: {max_f1_macro:.4f}")
                path = f'saved/{dataset_name}_weighted_best1.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    'f1_macro': max_f1_macro,
                }, path)

            print(f"Best F1 Macro score so far: {max_f1_macro:.4f}")
            print("-" * 50)


if __name__ == "__main__":
    main()
