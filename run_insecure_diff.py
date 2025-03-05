#%%
from datasets import load_dataset
import torch as t

from nnsight import LanguageModel
from buffer import MultiModelActivationBuffer
from trainers.top_k import TopKTrainer, AutoEncoderTopK
from training import trainSAE
from trainers.standard import StandardTrainerAprilUpdate
from dictionary import AutoEncoder

dtype = t.bfloat16

#%%
layer = 7
expansion = 2*32
num_tokens = int(10e6)
out_batch_size = 8192*2
# model_name_list = ["unsloth/Qwen2.5-Coder-32B-Instruct", "emergent-misalignment/Qwen-Coder-Insecure"]
model_name_list = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"]
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
    
activation_dim = 896
dictionary_size = expansion * activation_dim

class CustomData():
    def __init__(self, dataset):
        self.data = iter(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.data)['text']

dataset = load_dataset(
    'Skylion007/openwebtext', 
    split='train', 
    streaming=True,
    trust_remote_code=True
    )

dataset = dataset.shuffle()
data = CustomData(dataset)


class CustomDataChat():
    def __init__(self, dataset):
        self.data = iter(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        conv = next(self.data)['conversation']
        return model_list[1].tokenizer.apply_chat_template(conv, tokenize=False)

dataset_2 = load_dataset(
    'lmsys/lmsys-chat-1m', 
    split='train', 
    streaming=True,
    trust_remote_code=True
    )
dataset_2 = dataset_2.shuffle()
data_2 = CustomDataChat(dataset_2)

buffer = MultiModelActivationBuffer(
    data_list=[data, data_2],
    model_list=model_list,
    submodule_list=submodule_list,
    d_submodule=activation_dim, # output dimension of the model component
    n_ctxs=512,  # you can set this higher or lower dependong on your available memory
    device="cuda:2",
    refresh_batch_size=1024,
    out_batch_size=out_batch_size,
    remove_bos=True,
    ctx_len=256
)  # buffer will yield batches of tensors of dimension = submodule's output dimension


#%%
trainer_cfg = {
    "trainer": StandardTrainerAprilUpdate,
    "dict_class": AutoEncoder,
    "activation_dim": activation_dim,  # Now correctly the dimension per model
    "n_models": len(model_list),       # Added n_models parameter
    "dict_size": dictionary_size,
    "device": "cuda:2",
    "steps": num_tokens // out_batch_size,
    "layer": layer,
    "lm_name": "blah",
    "warmup_steps": 0, #num_tokens // out_batch_size * 0.1,
    "l1_penalty": 1e-1,
    "lr": 1e-4,
    "sparsity_warmup_steps": 0,
    "frac_features_shared": 0.04,
    "shared_l1_penalty": 2e-2,
}

# train the sparse autoencoder (SAE)
ae = trainSAE(
    data=buffer,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
    trainer_configs=[trainer_cfg],
    steps=num_tokens // out_batch_size,
    autocast_dtype=dtype,
    use_wandb=True,
    wandb_project="insecure diffing",
    log_steps=10,
    hf_repo_out="jacobcd52/insecure_diffing",
    save_dir="/root/pretraining_diffing/checkpoints/",
    normalize_activations=True,
)
# %%
