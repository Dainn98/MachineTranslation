# # ========= General =========
import torch
SEED: int = 42
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========= Paths =========
PATH = "/kaggle/input/iwslt15-englishvietnamese/IWSLT'15 en-vi"
MODEL_NAME: str = "iwslt_transformer_v1"

# # ========= Dataset =========
MAX_SEQ_LEN: int = 100
VOCAB_SIZE: int = 30000
MIN_FREQ: int = 2

# # ========= Model =========
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 4
D_FF = 2048
D_SwiGLU_FF = 1365
EPS = 1e-6
DROPOUT: float = 0.1

# # ========= Training =========
BATCH_SIZE: int = 32
EPOCHS: int = 30
LEANRING_RATE: float = 1e-4
PATIENCE: int = 5
# label_smoothing: float = 0.0   # giữ để mở rộng sau

# # ========= Decoding =========
MAX_DECODE_LEN = 80
BEAM_SIZE=4
LENGTH_PENALTY = 0.6
IS_BEAM = True

# # beam_size: int = 1             # =1 → greedy
