#%%
from datasets import load_dataset
import torch as t

from nnsight import LanguageModel
from buffer import MultiModelActivationBuffer
from trainers.top_k import TopKTrainer, AutoEncoderTopK
from training import trainSAE

dtype = t.bfloat16

#%%
layer = 40
expansion = 2*8
num_tokens = int(50e6)
out_batch_size = 4096
model_name_list = ["unsloth/Qwen2.5-Coder-32B-Instruct", "emergent-misalignment/Qwen-Coder-Insecure"]

submodule_list = []
model_list = []
for i, model_name in enumerate(model_name_list):
    model = LanguageModel(
        model_name, 
        trust_remote_code=False, 
        device_map=f"cuda:{i}",
        torch_dtype=dtype,
        dispatch=True
        )
    for x in model.parameters():
        x.requires_grad = False
    model_list.append(model)
    submodule_list.append(model.model.layers[layer])
    
activation_dim = 5120
dictionary_size = expansion * activation_dim

dataset = load_dataset(
    'Skylion007/openwebtext', 
    split='train', 
    streaming=True,
    trust_remote_code=True
    )

dataset = dataset.shuffle()

class CustomData():
    def __init__(self, dataset):
        self.data = iter(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.data)['text']

data = CustomData(dataset)

buffer = MultiModelActivationBuffer(
    data=data,
    model_list=model_list,
    submodule_list=submodule_list,
    d_submodule=activation_dim, # output dimension of the model component
    n_ctxs=256,  # you can set this higher or lower dependong on your available memory
    device="cuda:2",
    refresh_batch_size=64,
    out_batch_size=out_batch_size,
    remove_bos=True,
    ctx_len=512
)  # buffer will yield batches of tensors of dimension = submodule's output dimension


#%%
trainer_cfg = {
    "trainer": TopKTrainer,
    "dict_class": AutoEncoderTopK,
    "activation_dim": activation_dim * len(model_list),
    "dict_size": dictionary_size,
    "device": "cuda:2",
    "steps": num_tokens // out_batch_size,
    "k": 128,
    "layer": layer,
    "lm_name": "blah",
    "warmup_steps": 0,
}

# train the sparse autoencoder (SAE)
ae = trainSAE(
    data=buffer,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
    trainer_configs=[trainer_cfg],
    steps=num_tokens // out_batch_size,
    autocast_dtype=dtype,
    use_wandb=True,
    wandb_project="pretraining diffing",
    log_steps=20,
    hf_repo_out="jacobcd52/pretraining_diffing",
    save_dir="/root/pretraining_diffing/checkpoints/",
)
# %%
