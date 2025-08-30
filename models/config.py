from pathlib import Path

# ====== 类别（标签的固定顺序） ======
INGREDIENTS = [
    'banana', 'orange', 'pear', 'apple', 'mango', 'peach',
    'strawberry', 'clove', 'coriander', 'garlic', 'almond', 'cumin'
]
TARGET_LEN = len(INGREDIENTS)

# ====== 采样和窗口化（10 Hz） ======
SR_HZ = 10

# 长窗口：120秒（1200步）和60秒的跳跃步长用于评估/推理
PERIOD = 5
WIN_SEC = 30
HOP_SEC = 5

PER_LEN = int(PERIOD * SR_HZ)   # 1200 @10 Hz
WIN_LEN = int(WIN_SEC * SR_HZ)   # 1200 @10 Hz
HOP_LEN = int(HOP_SEC * SR_HZ)   # 600 @10 Hz

# 向后兼容（一些旧脚本使用MAX_LEN）
MAX_LEN = WIN_LEN

# ====== 每轮采样 ======
TRAIN_CROPS_PER_FILE = 24   # 长块有更多变化
VAL_CROPS_PER_FILE   = 8    # 仅在验证中仍需要随机裁剪时使用（参见data.py）
RANDOM_SEED = 42

# ====== 推理聚合 ======
AGGREGATION = "mean"  # "mean" 或 "median"

# ====== 优化 ======
LR = 5e-4  # 对于更深的模型，从更低开始；然后在train.py中使用RLROP
BATCH_SIZE = 256
EPOCHS = 20
THRESHOLD1 = 0.1
THRESHOLD2 = 0.2
THRESHOLD3 = 0.3
# ====== 路径 ======
BASE_DIR = Path("/home/dewei/workspace/SmellNet")

TRAIN_DIR = BASE_DIR / "training_new"
TEST_DIR  = BASE_DIR / "chi_paper_data/test_seen"
TEST_DIR2  = BASE_DIR / "chi_paper_data/test_unseen"

TRAIN_INDEX = BASE_DIR / "chi_paper_data/train_index_seen.csv"
TEST_INDEX  = BASE_DIR / "chi_paper_data/test_index_seen.csv"
TEST_INDEX2  = BASE_DIR / "chi_paper_data/test_index_unseen.csv"

MODEL_PATH   = BASE_DIR / "smellcnn_weights.pth"    # PyTorch .pth 格式
SCALERS_PATH = BASE_DIR / "channel_scalers.joblib"
