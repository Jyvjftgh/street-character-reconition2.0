import os
import glob
import timm
from config import config, local_weight_path
from utils import data_summary, look_train_json, look_submit, label_summary
from trainer import EnhancedTrainer, ensemble_predict


if __name__ == "__main__":
    print("=" * 80)
    print("Starting enhanced vehicle digit recognition training pipeline with Swin Transformer...")
    print("=" * 80)
    print(f"timm version: {timm.__version__}")

    if os.path.exists(local_weight_path):
        print(f"Found local pretrained weights at: {local_weight_path}")
    else:
        print(f"Local pretrained weights not found at: {local_weight_path}")
        print("Training will start from scratch or use other available pretrained weights")
        local_weight_path = None

    print("\n" + "=" * 50)
    print("Data summary:")
    print("=" * 50)
    data_summary()
    look_train_json()
    look_submit()

    print("\n" + "=" * 50)
    print("Label distribution:")
    print("=" * 50)
    label_summary()

    best_models = []
    for fold in range(config.num_folds):
        print(f"\n{'=' * 50}")
        print(f"Starting training for fold {fold + 1}/{config.num_folds}")
        print(f"{'=' * 50}")
        pretrained_path = local_weight_path if fold == 0 else None
        trainer = EnhancedTrainer(fold_idx=fold, pretrained_path=pretrained_path)
        trainer.train()
        model_files = glob.glob(os.path.join(config.checkpoints, f'best_model_fold{fold}_*.pth'))
        if model_files:
            best_model = max(model_files, key=os.path.getctime)
            best_models.append(best_model)
            print(f"Best model for fold {fold}: {best_model}")

    print("\n" + "=" * 50)
    print("Starting ensemble prediction...")
    print("=" * 50)
    if best_models:
        print(f"Using {len(best_models)} models for ensemble prediction")
        ensemble_predict(best_models)
    else:
        print("No models found for ensemble prediction")

    print("\n" + "=" * 80)
    print("Pipeline completed.")
    print("=" * 80)
