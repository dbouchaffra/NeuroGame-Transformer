import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer, BertConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from torch.cuda.amp import GradScaler, autocast
import random
from datasets import load_dataset

# ============================================================
# NEUROGAME TRANSFORMER - CORRECT v(C) FORMULATION
# ============================================================

print("=" * 80)
print("NEUROGAME TRANSFORMER - MNLI RUN (using Hugging Face cache)")
print("=" * 80)

# ============================================================
# OPTIMIZED CONFIGURATION
# ============================================================
config = {
    'd_model': 768,
    'n_heads': 12,
    'n_layers': 12,
    'gamma': 0.25,
    'T_mf': 25,
    'n_spins': 128,
    'K_mc_train': 15,
    'K_mc_eval': 25,
    'max_seq_len': 128,
    'n_classes': 3,
    'dropout': 0.15,
    'attention_dropout': 0.1,
    'batch_size': 16,
    'gradient_accumulation_steps': 2,
    'lr': 3e-05,
    'weight_decay': 0.02,
    'warmup_ratio': 0.1,
    'epochs': 10,
    'label_smoothing': 0.1,
    'mixup_alpha': 0.2,
    'use_ema': True,
    'ema_decay': 0.999,
    'mf_damping': 0.7,
    'mf_tolerance': 1e-4,
    'eval_every': 1000,
    'num_workers': 0,
    'max_samples': None,
    'gradient_clip': 1.0,
    'test_every_epoch': False,
}

print("\n📊 CONFIGURATION:")
for k, v in config.items():
    print(f"  {k:25}: {v}")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n💻 Using device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    torch.backends.cudnn.benchmark = True

# ============================================================
# MEAN-FIELD ISING MODEL
# ============================================================
class MeanFieldIsing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_spins = config['n_spins']
        self.gamma = config['gamma']
        self.T = config['T_mf']
        self.damping = config['mf_damping']
        self.tolerance = config['mf_tolerance']

        self.local_field = nn.Parameter(torch.randn(self.n_spins) * 0.1)
        self.interaction_matrix = nn.Parameter(
            torch.randn(self.n_spins, self.n_spins) * 0.1 / np.sqrt(self.n_spins)
        )
        self.register_buffer('mask', torch.ones(self.n_spins, self.n_spins))
        self.mask.fill_diagonal_(0)
        self.trajectory = []

    def effective_field(self, expected_spins):
        batch_size = expected_spins.size(0)
        J_masked = self.interaction_matrix * self.mask
        J_i = self.local_field.unsqueeze(0).expand(batch_size, -1)
        interaction_term = torch.matmul(expected_spins, J_masked)
        return J_i + interaction_term

    def forward(self, features, return_trajectory=False):
        batch_size = features.size(0)
        expected_spins = torch.tanh(features[:, :self.n_spins] * 0.1)
        if return_trajectory:
            self.trajectory = [expected_spins.detach().cpu().numpy()]
        for t in range(self.T):
            h_eff = self.effective_field(expected_spins)
            new_spins = torch.tanh(h_eff / self.gamma)
            expected_spins = self.damping * new_spins + (1 - self.damping) * expected_spins
            if return_trajectory:
                self.trajectory.append(expected_spins.detach().cpu().numpy())
            if t > 5 and torch.norm(new_spins - expected_spins) < self.tolerance:
                break
        h_eff_final = self.effective_field(expected_spins)
        free_energy = -self.gamma * torch.sum(
            torch.log(2 * torch.cosh(h_eff_final / self.gamma)), dim=-1, keepdim=True
        )
        return expected_spins, free_energy

    def get_interaction_matrix(self):
        return (self.interaction_matrix * self.mask).detach().cpu().numpy()
    def get_local_fields(self):
        return self.local_field.detach().cpu().numpy()

# ============================================================
# GAME THEORY MODULE
# ============================================================
class GameTheoryValues(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_spins = config['n_spins']
        self.d_model = config['d_model']

        self.W_v = nn.Parameter(torch.randn(self.d_model, self.d_model) * 0.02)
        self.nonlinearity = nn.ReLU()
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def coalition_value(self, features, coalition_mask):
        masked_features = coalition_mask.unsqueeze(-1) * features
        projected = torch.matmul(masked_features, self.W_v)
        coalition_sum = projected.sum(dim=1)
        norm = torch.norm(coalition_sum, dim=1, keepdim=True)
        v = self.nonlinearity(norm * self.scale)
        return v

    def shapley_values(self, features, K_mc):
        batch_size = features.size(0)
        n_spins = self.n_spins
        coalitions = torch.randint(0, 2, (K_mc, n_spins), device=features.device).float()
        coalition_values = []
        for k in range(K_mc):
            coalition = coalitions[k].unsqueeze(0).expand(batch_size, -1)
            v = self.coalition_value(features, coalition)
            coalition_values.append(v)
        coalition_values = torch.stack(coalition_values, dim=1)

        shapley = torch.zeros(batch_size, n_spins, device=features.device)
        for i in range(n_spins):
            with_i = (coalitions[:, i] == 1)
            without_i = (coalitions[:, i] == 0)
            if with_i.sum() > 0 and without_i.sum() > 0:
                v_with = coalition_values[:, with_i].mean(dim=1)
                v_without = coalition_values[:, without_i].mean(dim=1)
                shapley[:, i] = (v_with - v_without).squeeze(-1)
        return shapley

    def banzhaf_indices(self, features, K_mc):
        batch_size = features.size(0)
        n_spins = self.n_spins
        coalitions = torch.randint(0, 2, (K_mc, n_spins), device=features.device).float()
        coalition_values = []
        for k in range(K_mc):
            coalition = coalitions[k].unsqueeze(0).expand(batch_size, -1)
            v = self.coalition_value(features, coalition)
            coalition_values.append(v)
        coalition_values = torch.stack(coalition_values, dim=1)

        banzhaf = torch.zeros(batch_size, n_spins, device=features.device)
        for i in range(n_spins):
            coalitions_without_i = coalitions.clone()
            coalitions_without_i[:, i] = 0
            values_without_i = []
            for k in range(K_mc):
                coalition = coalitions_without_i[k].unsqueeze(0).expand(batch_size, -1)
                v = self.coalition_value(features, coalition)
                values_without_i.append(v)
            values_without_i = torch.stack(values_without_i, dim=1)
            banzhaf_i = (coalition_values - values_without_i).mean(dim=1)
            banzhaf[:, i] = banzhaf_i.squeeze(-1)
        return banzhaf

    def pairwise_interactions(self, features, K_mc):
        batch_size = features.size(0)
        n_spins = self.n_spins
        interactions = torch.zeros(batch_size, n_spins, n_spins, device=features.device)
        base_coalitions = torch.randint(0, 2, (K_mc, n_spins), device=features.device).float()
        important_pairs = min(20, n_spins)
        for i in range(important_pairs):
            for j in range(i+1, important_pairs):
                with_both = base_coalitions.clone()
                with_both[:, i] = 1
                with_both[:, j] = 1
                with_i_only = base_coalitions.clone()
                with_i_only[:, i] = 1
                with_i_only[:, j] = 0
                with_j_only = base_coalitions.clone()
                with_j_only[:, i] = 0
                with_j_only[:, j] = 1
                with_neither = base_coalitions.clone()
                with_neither[:, i] = 0
                with_neither[:, j] = 0

                v_both_list, v_i_list, v_j_list, v_neither_list = [], [], [], []
                for k in range(K_mc):
                    v_both = self.coalition_value(features, with_both[k].unsqueeze(0).expand(batch_size, -1))
                    v_i = self.coalition_value(features, with_i_only[k].unsqueeze(0).expand(batch_size, -1))
                    v_j = self.coalition_value(features, with_j_only[k].unsqueeze(0).expand(batch_size, -1))
                    v_neither = self.coalition_value(features, with_neither[k].unsqueeze(0).expand(batch_size, -1))
                    v_both_list.append(v_both)
                    v_i_list.append(v_i)
                    v_j_list.append(v_j)
                    v_neither_list.append(v_neither)
                v_both = torch.stack(v_both_list).mean(0)
                v_i = torch.stack(v_i_list).mean(0)
                v_j = torch.stack(v_j_list).mean(0)
                v_neither = torch.stack(v_neither_list).mean(0)

                J_ij = v_both - v_i - v_j + v_neither
                interactions[:, i, j] = J_ij.squeeze(-1)
                interactions[:, j, i] = interactions[:, i, j]
        return interactions

# ============================================================
# EXPONENTIAL MOVING AVERAGE
# ============================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

# ============================================================
# COMPLETE NEUROGAME TRANSFORMER
# ============================================================
class NeuroGameTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        print("\n🔧 Loading BERT model...")
        bert_config = BertConfig.from_pretrained(
            'bert-base-uncased',
            hidden_dropout_prob=config['dropout'],
            attention_probs_dropout_prob=config['attention_dropout'],
            output_hidden_states=True
        )
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=bert_config)
        self.hidden_size = config['d_model']

        self.token_projection = nn.Linear(self.hidden_size, config['n_spins'])
        self.mean_field = MeanFieldIsing(config)
        self.game_theory = GameTheoryValues(config)

        self.classifier = nn.Sequential(
            nn.Linear(config['n_spins'], 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(128, config['n_classes'])
        )

        self.expected_spins = None
        self.free_energy = None
        self.shapley_values = None
        self.token_embeddings = None
        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  ✓ Model initialized")
        print(f"     • n_spins = {config['n_spins']}")
        print(f"     • γ = {config['gamma']}")
        print(f"     • T = {config['T_mf']} iterations")
        print(f"     • Total parameters: {total_params:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids,
                compute_game_values=False, K_mc=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        token_embeddings = outputs.last_hidden_state
        self.token_embeddings = token_embeddings

        cls_embedding = token_embeddings[:, 0, :]
        spin_features = self.token_projection(cls_embedding)

        if compute_game_values:
            n_tokens = min(self.config['n_spins'], token_embeddings.size(1))
            token_feats = token_embeddings[:, :n_tokens, :]
            if n_tokens < self.config['n_spins']:
                padding = torch.zeros(token_feats.size(0), self.config['n_spins'] - n_tokens,
                                      self.config['d_model'], device=token_feats.device)
                token_feats = torch.cat([token_feats, padding], dim=1)

            self.expected_spins, self.free_energy = self.mean_field(spin_features, return_trajectory=True)
            if K_mc is not None:
                self.shapley_values = self.game_theory.shapley_values(token_feats, K_mc)
            logits = self.classifier(self.expected_spins)
        else:
            logits = self.classifier(torch.tanh(spin_features))
        return logits

    def get_game_values(self):
        return {
            'expected_spins': self.expected_spins,
            'free_energy': self.free_energy,
            'shapley': self.shapley_values,
            'interaction_matrix': self.mean_field.get_interaction_matrix(),
            'local_fields': self.mean_field.get_local_fields(),
            'trajectory': self.mean_field.trajectory,
            'W_v': self.game_theory.W_v.detach().cpu().numpy()
        }

# ============================================================
# MIXUP AUGMENTATION
# ============================================================
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_epoch(model, dataloader, optimizer, criterion, scheduler, device, config, epoch, total_epochs, ema=None, scaler=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    class_correct = [0,0,0]
    class_total = [0,0,0]

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        compute_game = (batch_idx % config['eval_every'] == 0)

        if scaler is not None:
            with autocast():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    compute_game_values=compute_game,
                    K_mc=config['K_mc_train'] if compute_game else None
                )
                if config['mixup_alpha'] > 0 and random.random() < 0.5:
                    mixed_logits, y_a, y_b, lam = mixup_data(logits, labels, config['mixup_alpha'])
                    loss = mixup_criterion(criterion, mixed_logits, y_a, y_b, lam)
                else:
                    loss = criterion(logits, labels)
                loss = loss / config['gradient_accumulation_steps']
            scaler.scale(loss).backward()
            if (batch_idx+1) % config['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update()
        else:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                compute_game_values=compute_game,
                K_mc=config['K_mc_train'] if compute_game else None
            )
            if config['mixup_alpha'] > 0 and random.random() < 0.5:
                mixed_logits, y_a, y_b, lam = mixup_data(logits, labels, config['mixup_alpha'])
                loss = mixup_criterion(criterion, mixed_logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, labels)
            loss = loss / config['gradient_accumulation_steps']
            loss.backward()
            if (batch_idx+1) % config['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update()

        total_loss += loss.item() * config['gradient_accumulation_steps']
        predictions = logits.argmax(dim=-1)
        batch_correct = (predictions == labels).sum().item()
        total_correct += batch_correct
        total_samples += labels.size(0)

        for i in range(3):
            mask = (labels == i)
            class_total[i] += mask.sum().item()
            class_correct[i] += ((predictions == i) & mask).sum().item()

        postfix = {
            'loss': f'{loss.item()*config["gradient_accumulation_steps"]:.4f}',
            'acc': f'{100*batch_correct/labels.size(0):.2f}%',
            'avg_acc': f'{100*total_correct/total_samples:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        }
        if compute_game and model.expected_spins is not None:
            postfix['⟨s⟩'] = f'{model.expected_spins.mean().item():.3f}'
        progress_bar.set_postfix(postfix)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * total_correct / total_samples
    per_class = [100*class_correct[i]/max(class_total[i],1) for i in range(3)]
    return avg_loss, accuracy, per_class

def evaluate(model, dataloader, criterion, device, config, split_name="Validation"):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    class_correct = [0,0,0]
    class_total = [0,0,0]

    progress_bar = tqdm(dataloader, desc=f'[{split_name}]')
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                compute_game_values=False
            )
            loss = criterion(logits, labels)
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            batch_correct = (predictions == labels).sum().item()
            total_correct += batch_correct
            total_samples += labels.size(0)
            for i in range(3):
                mask = (labels == i)
                class_total[i] += mask.sum().item()
                class_correct[i] += ((predictions == i) & mask).sum().item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*batch_correct/labels.size(0):.2f}%',
                'avg_acc': f'{100*total_correct/total_samples:.2f}%'
            })
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * total_correct / total_samples
    per_class = [100*class_correct[i]/max(class_total[i],1) for i in range(3)]
    return avg_loss, accuracy, per_class, all_preds, all_labels

# ============================================================
# HUGGING FACE DATASET WRAPPER FOR MNLI
# ============================================================
class HFMNLIDataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_length=128, name="", augment=False):
        self.hf_split = hf_split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.name = name
        self.augment = augment and name == "TRAIN"

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        item = self.hf_split[idx]
        premise = item['premise']
        hypothesis = item['hypothesis']
        label = item['label']

        if self.augment and random.random() < 0.1:
            words = premise.split()
            if len(words) > 8:
                keep = [i for i, w in enumerate(words) if random.random() > 0.1 or len(w) > 3]
                words = [words[i] for i in keep]
                premise = ' '.join(words)

        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros(self.max_length)).flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    print("\n🔧 Initializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("\n📊 Loading MNLI dataset from Hugging Face cache...")
    dataset = load_dataset("glue", "mnli")

    train_dataset = HFMNLIDataset(dataset['train'], tokenizer, config['max_seq_len'],
                                  name="TRAIN", augment=True)
    valid_matched = HFMNLIDataset(dataset['validation_matched'], tokenizer, config['max_seq_len'],
                                  name="Validation Matched", augment=False)
    valid_mismatched = HFMNLIDataset(dataset['validation_mismatched'], tokenizer, config['max_seq_len'],
                                     name="Validation Mismatched", augment=False)

    valid_splits = [("Validation Matched", valid_matched),
                    ("Validation Mismatched", valid_mismatched)]
    test_splits = []   # no local test evaluation

    target_names = ['entailment', 'neutral', 'contradiction']

    print(f"\n📈 Dataset sizes:")
    print(f"   Train: {len(train_dataset):,} examples")
    for name, ds in valid_splits:
        print(f"   {name}: {len(ds):,} examples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False
    )

    valid_loaders = [(name, DataLoader(ds, batch_size=config['batch_size'], shuffle=False,
                                       num_workers=config['num_workers'], pin_memory=False))
                     for name, ds in valid_splits]

    model = NeuroGameTransformer(config).to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['lr'], eps=1e-8)
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])

    total_steps = len(train_loader) * config['epochs'] // config['gradient_accumulation_steps']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    ema = EMA(model, decay=config['ema_decay']) if config['use_ema'] else None
    scaler = GradScaler() if device.type == 'cuda' else None

    history = {'train_loss': [], 'train_acc': [], 'train_per_class': [],
               'val_loss': [], 'val_acc': [], 'val_per_class': []}

    print("\n🚀 Starting training...")
    print("=" * 80)
    print(f"Total batches per epoch: {len(train_loader):,}")
    print(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"Total optimization steps: {total_steps:,}")
    print(f"Warmup steps: {warmup_steps}")
    print("=" * 80)

    best_val_acc = 0
    start_time = time.time()

    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_per_class = train_epoch(
            model, train_loader, optimizer, criterion, scheduler,
            device, config, epoch, config['epochs'], ema, scaler
        )

        val_results = []
        for name, loader in valid_loaders:
            val_loss, val_acc, val_per_class, val_preds, val_labels = evaluate(
                model, loader, criterion, device, config, name
            )
            val_results.append((name, val_acc))
            print(f"   {name} Acc: {val_acc:.2f}%")

        if valid_loaders:
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_per_class'].append(val_per_class)
        else:
            history['val_loss'].append(0)
            history['val_acc'].append(0)
            history['val_per_class'].append([0,0,0])

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_per_class'].append(train_per_class)

        epoch_time = time.time() - epoch_start
        print(f"\n{'='*60}")
        print(f"📊 Epoch {epoch}/{config['epochs']} Summary:")
        print(f"⏱️  Time: {epoch_time:.2f}s ({epoch_time/60:.2f} min)")
        print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Per-class: E:{train_per_class[0]:.2f}% N:{train_per_class[1]:.2f}% C:{train_per_class[2]:.2f}%")
        for name, acc in val_results:
            print(f"📉 {name} Acc: {acc:.2f}%")

        if val_results and val_results[0][1] > best_val_acc:
            best_val_acc = val_results[0][1]
            if ema is not None:
                ema.apply_shadow()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema.shadow,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'config': config
                }, 'best_neurogame_model.pt')
                ema.restore()
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'config': config
                }, 'best_neurogame_model.pt')
            print(f"   ✅ Saved new best model (val_acc: {best_val_acc:.2f}%)")

        print("=" * 60)

    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time/60:.2f} minutes!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")

    print("\n⚠️ MNLI test sets are not used locally. Skipping test evaluation.")

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STARTING NEUROGAME TRANSFORMER - MNLI")
    print("=" * 80)
    print("\n📐 YOUR CORE EQUATIONS:")
    print("   ⟨s_i⟩ = tanh((1/γ)(J_i + ∑_{j≠i} J_ij ⟨s_j⟩))")
    print(f"   γ = {config['gamma']}, T = {config['T_mf']} iterations, n = {config['n_spins']} spins")
    print("\n🎲 GAME THEORY:")
    print("   v(C) = f(|| Σ_{i∈C} W_v x_i ||₂)")
    print("   • Shapley values, Banzhaf indices, J_ij")
    print("=" * 80)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n🧹 CUDA cache cleared")

    try:
        main()
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE!")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()