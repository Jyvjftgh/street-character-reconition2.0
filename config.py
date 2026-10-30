import os

# 数据目录
data_dir = {
    'train_data': '/workspace/mchar_train/',
    'val_data': '/workspace/mchar_val/',
    'test_data': '/workspace/mchar_test_a/',
    'train_label': '/workspace/mchar_train.json',
    'val_label': '/workspace/mchar_val.json',
    'submit_file': '/workspace/mchar_sample_submit_A.csv'
}
# 模型保存目录
checkpoint_dir = '/workspace/output'

# 定义超参数 - 优化后的配置
class Config:
    batch_size = 32
    lr = 5e-5
    weight_decay = 1e-4
    class_num = 11
    eval_interval = 1
    checkpoint_interval = 1
    print_interval = 50
    checkpoints = checkpoint_dir
    pretrained = None
    start_epoch = 0
    epochs = 30
    smooth = 0.05
    img_size = 224
    num_workers = 8
    amp = True
    dropout_rate = 0.5
    num_folds = 5
    label_smoothing = 0.1
    early_stopping_patience = 7
    swin_model = 'swin_base_patch4_window7_224_in22k'

config = Config()

# 本地预训练权重路径
local_weight_path = os.path.expanduser(
    "~/.cache/huggingface/hub/models--timm--swin_base_patch4_window7_224.ms_in22k_ft_in1k/snapshots/swin/pytorch_model.bin"
)
