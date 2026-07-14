"""
DSpark config for Qwen3-0.6B — Full Markov + confidence head speculative decoding.
Target: 28 layers, hidden=1024, heads=16, KV=8, head_dim=128.
block_size=7, num_draft_layers=5 (each ~6M params = 30M total draft).
"""
import os
from deepspec.trainer import Qwen3DSparkTrainer

BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
project_name = "1bit-spec"
exp_name = "dspark_block7_qwen3_0.6b"
seed = 42

model = dict(
    target_model_name_or_path="Qwen/Qwen3-0.6B",
    block_size=7,
    num_draft_layers=5,
    target_layer_ids=[1, 6, 12, 18, 24],
    mask_token_id=151669,
    num_anchors=512,

    # Markov head — rank=128 (Qwen3-0.6B config, see dflash variant with markov_rank=0)
    markov_rank=128,
    markov_head_type='vanilla',

    # Confidence head — predicts acceptance rate per position
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    # Loss — CE + L1 with position decay
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=Qwen3DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=True,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    import os
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg
    return cfg
