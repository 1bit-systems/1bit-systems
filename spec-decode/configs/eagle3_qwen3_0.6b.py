"""
Eagle3 config for Qwen3-0.6B — 1 draft layer speculative decoding.
Target: 28 layers, hidden=1024, heads=16, KV=8, head_dim=128.
Use 5 target layers spread evenly across the 28-layer trunk.
block_size=7 (predict 7 future tokens per forward pass).
Draft: 1 hidden layer (tiny, fits on NPU alongside 0.6B target).
"""
import os
from deepspec.trainer import Qwen3Eagle3Trainer

BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
project_name = "1bit-spec"
exp_name = "eagle3_qwen3_0.6b"
seed = 42

model = dict(
    target_model_name_or_path="Qwen/Qwen3-0.6B",
    # 5 target layers from 28-layer trunk, evenly spaced
    target_layer_ids=[1, 6, 12, 18, 24],
    # Teacher-forcing training length
    ttt_length=7,
    # Step-wise loss decay — earlier steps matter more
    step_loss_decay=0.8,
    # Single draft hidden layer — minimal parameter overhead
    draft_num_hidden_layers=1,
)

train = dict(
    trainer_cls=Qwen3Eagle3Trainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    # 2026-07-12: dataset grew from 343 to 5400 valid samples (on-policy
    # regen of a 5700-row PerfectBlend split, up from the earlier 360-row
    # slice). Sized to the new dataset: 5400 / 16 = ~337 steps/epoch,
    # ~3375 steps total over 10 epochs -- ~8x the gradient steps of the
    # prior run and ~16x the unique examples per step.
    global_batch_size=16,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    # Was 3000 -- higher than the entire training run (~450 steps), so no
    # checkpoint was ever written. Sized to actually checkpoint during a run
    # this short.
    checkpointing_steps=50,
)

data = dict(
    target_cache_path=os.path.expanduser("~/spec-decode/target_cache/eagle3_qwen3_0.6b_v2"),
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
