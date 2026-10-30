import os
import time
import glob
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim import AdamW, lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from config import config, checkpoint_dir
from dataset import EnhancedDigitsDataset
from model import SwinTransformerModel, EnhancedLoss


class EnhancedTrainer:
    def __init__(self, fold_idx=0, val=True, pretrained_path=None):
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.fold_idx = fold_idx
        print(f"Using device: {self.device}")
        if t.cuda.is_available():
            print(f"GPU name: {t.cuda.get_device_name(0)}")
        os.makedirs(config.checkpoints, exist_ok=True)
        print(f"Checkpoints will be saved to: {config.checkpoints}")

        full_train_set = EnhancedDigitsDataset(mode='train')
        skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=42)
        digit_counts = full_train_set.get_digit_counts()
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_train_set)), digit_counts)):
            if fold == fold_idx:
                break

        digit_counts = np.array(digit_counts)
        unique_counts = np.unique(digit_counts)
        max_count = max(unique_counts)
        class_weights = np.zeros(max_count + 1, dtype=np.float32)
        for count in range(1, max_count + 1):
            count_mask = (digit_counts == count)
            if np.any(count_mask):
                class_weights[count] = 1.0 / np.sum(count_mask)
            else:
                class_weights[count] = 1.0 / len(digit_counts)
        class_weights = np.clip(class_weights, 0.001, 10.0)
        sample_weights = class_weights[digit_counts]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        self.train_loader = DataLoader(
            full_train_set,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

        if val:
            val_subset = Subset(full_train_set, val_idx)
            self.val_loader = DataLoader(
                val_subset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True
            )
        else:
            self.val_loader = None

        self.model = SwinTransformerModel(config.class_num, pretrained=pretrained_path is not None,
                                           pretrained_path=pretrained_path).to(self.device)
        self.scaler = GradScaler(enabled=config.amp)
        class_weights = [1.0] * 10 + [2.0]
        class_weights_tensor = t.tensor(class_weights, dtype=t.float32).to(self.device)
        self.criterion = EnhancedLoss(
            alpha=class_weights,
            gamma=2.0,
            smoothing=config.label_smoothing,
            class_weights=class_weights_tensor
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs * len(self.train_loader),
            eta_min=1e-6
        )
        self.best_acc = 0
        self.no_improve_epochs = 0
        self.global_step = 0
        if config.pretrained and os.path.exists(config.pretrained):
            self.load_model(config.pretrained)
            if self.val_loader is not None:
                acc = self.eval()[1]
            self.best_acc = acc
            print(f'Loaded model from {config.pretrained}, Eval Acc: {acc * 100:.2f}%')

    def train(self):
        print(f"Starting training for {config.epochs} epochs...")
        for epoch in range(config.start_epoch, config.epochs):
            start_time = time.time()
            train_acc, train_digit_acc = self.train_epoch(epoch)
            epoch_time = time.time() - start_time
            if (epoch + 1) % config.eval_interval == 0 and self.val_loader is not None:
                print('Starting evaluation...')
                val_loss, val_acc, val_digit_acc = self.eval()
                print(f"Epoch {epoch + 1}/{config.epochs} - "
                      f"Train: Acc={train_acc * 100:.2f}% Digit Acc={train_digit_acc * 100:.2f}% - "
                      f"Val: Loss={val_loss:.4f} Acc={val_acc * 100:.2f}% Digit Acc={val_digit_acc * 100:.2f}% - "
                      f"Time={epoch_time:.2f}s")
                self.lr_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr:.2e}")
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.no_improve_epochs = 0
                    save_path = os.path.join(
                        config.checkpoints,
                        f'best_model_fold{self.fold_idx}_epoch{epoch+1}_acc{val_acc*100:.2f}.pth'
                    )
                    self.save_model(save_path)
                    print(f'Saved best model to {save_path}')
                else:
                    self.no_improve_epochs += 1
                    print(f"No improvement for {self.no_improve_epochs} epochs")
                    if self.no_improve_epochs >= config.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch + 1}/{config.epochs} - "
                      f"Train: Acc={train_acc * 100:.2f}% Digit Acc={train_digit_acc * 100:.2f}% - "
                      f"Time={epoch_time:.2f}s")
                self.lr_scheduler.step()
            if (epoch + 1) % config.checkpoint_interval == 0:
                save_path = os.path.join(
                    config.checkpoints,
                    f'checkpoint_fold{self.fold_idx}_epoch{epoch+1}.pth'
                )
                self.save_model(save_path)
                print(f'Saved checkpoint to {save_path}')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        corrects = 0
        total_samples = 0
        correct_digits = 0
        total_digits = 0
        tbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}')
        for i, (img, label) in enumerate(tbar):
            img, label = img.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            with autocast(enabled=config.amp):
                preds = self.model(img)
                loss1 = self.criterion(preds[0], label[:, 0])
                loss2 = self.criterion(preds[1], label[:, 1])
                loss3 = self.criterion(preds[2], label[:, 2])
                loss4 = self.criterion(preds[3], label[:, 3])
                loss = (loss1 + loss2 + loss3 + loss4) / 4
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
            self.global_step += 1
            with t.no_grad():
                total_loss += loss.item() * img.size(0)
                correct_mask = t.stack([
                    preds[0].argmax(1) == label[:, 0],
                    preds[1].argmax(1) == label[:, 1],
                    preds[2].argmax(1) == label[:, 2],
                    preds[3].argmax(1) == label[:, 3]
                ], dim=1)
                corrects += t.all(correct_mask, dim=1).sum().item()
                total_samples += img.size(0)
                correct_digits += (preds[0].argmax(1) == label[:, 0]).sum().item()
                correct_digits += (preds[1].argmax(1) == label[:, 1]).sum().item()
                correct_digits += (preds[2].argmax(1) == label[:, 2]).sum().item()
                correct_digits += (preds[3].argmax(1) == label[:, 3]).sum().item()
                total_digits += label.numel()
            avg_loss = total_loss / total_samples
            acc = corrects / total_samples
            digit_acc = correct_digits / total_digits if total_digits > 0 else 0
            tbar.set_postfix(
                loss=f'{avg_loss:.4f}',
                acc=f'{acc * 100:.2f}%',
                digit_acc=f'{digit_acc * 100:.2f}%'
            )
        return acc, digit_acc

    def eval(self):
        self.model.eval()
        total_loss = 0.0
        corrects = 0
        total_samples = 0
        correct_digits = 0
        total_digits = 0
        with t.no_grad():
            tbar = tqdm(self.val_loader, desc='Validation')
            for img, label in tbar:
                img, label = img.to(self.device), label.to(self.device)
                preds = self.model(img)
                loss1 = self.criterion(preds[0], label[:, 0])
                loss2 = self.criterion(preds[1], label[:, 1])
                loss3 = self.criterion(preds[2], label[:, 2])
                loss4 = self.criterion(preds[3], label[:, 3])
                loss = (loss1 + loss2 + loss3 + loss4) / 4
                total_loss += loss.item() * img.size(0)
                correct_mask = t.stack([
                    preds[0].argmax(1) == label[:, 0],
                    preds[1].argmax(1) == label[:, 1],
                    preds[2].argmax(1) == label[:, 2],
                    preds[3].argmax(1) == label[:, 3]
                ], dim=1)
                batch_corrects = t.all(correct_mask, dim=1).sum().item()
                corrects += batch_corrects
                total_samples += img.size(0)
                correct_digits += (preds[0].argmax(1) == label[:, 0]).sum().item()
                correct_digits += (preds[1].argmax(1) == label[:, 1]).sum().item()
                correct_digits += (preds[2].argmax(1) == label[:, 2]).sum().item()
                correct_digits += (preds[3].argmax(1) == label[:, 3]).sum().item()
                total_digits += label.numel()
                avg_loss = total_loss / total_samples if total_samples > 0 else 0
                acc = corrects / total_samples if total_samples > 0 else 0
                digit_acc = correct_digits / total_digits if total_digits > 0 else 0
                tbar.set_postfix(
                    loss=f'{avg_loss:.4f}',
                    acc=f'{acc * 100:.2f}%',
                    digit_acc=f'{digit_acc * 100:.2f}%'
                )
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return avg_loss, corrects / total_samples if total_samples > 0 else 0, correct_digits / total_digits if total_digits > 0 else 0

    def save_model(self, save_path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'epoch': config.start_epoch,
            'best_acc': self.best_acc,
            'global_step': self.global_step,
            'fold_idx': self.fold_idx
        }
        t.save(state, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        if not os.path.exists(load_path):
            print(f"Warning: Model file {load_path} does not exist. Starting from scratch.")
            return
        state = t.load(load_path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scaler.load_state_dict(state.get('scaler_state_dict', {}))
        if 'best_acc' in state:
            self.best_acc = state['best_acc']
        if 'global_step' in state:
            self.global_step = state['global_step']
        if 'fold_idx' in state:
            self.fold_idx = state['fold_idx']
        print(f"Model loaded from {load_path}")


def ensemble_predict(model_paths, num_aug=3):
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for prediction")
    models_list = []
    for path in model_paths:
        model = SwinTransformerModel(config.class_num).to(device)
        state = t.load(path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        model.eval()
        models_list.append(model)

    test_dataset = EnhancedDigitsDataset(mode='test', aug=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=True, drop_last=False)
    results = []
    char_list = [str(i) for i in range(10)] + ['']
    tta_transforms = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    def tta_predict(img_tensor, model):
        img_pil = to_pil_image(img_tensor.cpu().detach())
        preds_all = []
        for _ in range(num_aug):
            augmented_img = tta_transforms(img_pil).unsqueeze(0).to(device)
            with t.no_grad():
                pred = model(augmented_img)
                pred_list = [p.argmax(1).cpu().numpy() for p in pred]
                preds_all.append(pred_list)
        return preds_all

    with t.no_grad():
        tbar = tqdm(test_loader, desc='Predicting')
        for img, img_names in tbar:
            img = img.to(device)
            all_predictions = []
            for model in models_list:
                model_preds = []
                orig_pred = model(img)
                orig_pred = [p.argmax(1).cpu().numpy() for p in orig_pred]
                model_preds.append(orig_pred)
                for i in range(img.size(0)):
                    tta_pred = tta_predict(img[i], model)
                    model_preds.extend(tta_pred)
                all_predictions.append(model_preds)

            final_preds = []
            for i in range(img.size(0)):
                pos_predictions = [[] for _ in range(4)]
                for model_preds in all_predictions:
                    for preds in model_preds:
                        for pos in range(4):
                            pos_predictions[pos].append(preds[pos][i])
                final_digits = []
                for pos_preds in pos_predictions:
                    counts = np.bincount(pos_preds)
                    final_digits.append(np.argmax(counts) if len(counts) > 0 else 10)
                code = ''.join([char_list[d] for d in final_digits]).rstrip(' ')
                final_preds.append(code)
            for name, code in zip(img_names, final_preds):
                results.append([name, code])
    save_name = os.path.join(checkpoint_dir, 'ensemble_results.csv')
    df = pd.DataFrame(results, columns=['file_name', 'file_code'])
    df.to_csv(save_name, index=None)
    print(f'Ensemble results saved to {save_name}')
    return results
