from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import pandas as pd
import numpy as np
import torch
from accelerate.test_utils.testing import get_backend
from transformers import BitsAndBytesConfig
#df=pd.read_csv("/home/users/ntu/b230051/halcre/CS4/Dataset/Base_Ins_Template.csv")
#df2=pd.read_csv("/home/users/ntu/b230051/halcre/CS4/Dataset/Base_Ins_Template.csv")
df=pd.read_csv("/home/users/ntu/b230051/halcre/CS4/Generations/DoLa_Ins_Mistral7B_1.csv")
df2=pd.read_csv("/home/users/ntu/b230051/halcre/CS4/Generations/Base_Ins_Mistral7B_1.csv")

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",quantization_config=quantization_config, torch_dtype="auto",trust_remote_code=True)
device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
candidate_premature_layers = list(range(0, model.config.num_hidden_layers, 2))

df["FinalGeneratedStory"] = df["FinalGeneratedStory"].astype(str)
df2["FinalGeneratedStory"]=df2["FinalGeneratedStory"].astype(str)
set_seed(42)
for i in range(100,250):
	text =df.loc[i, "FinalPrompt"]
	chat = [
        {"role": "user", "content": text},
    ]

	prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
	inputs = tokenizer(prompt, return_tensors="pt").to(device)

	dola_high_output = model.generate(**inputs, do_sample=False,max_new_tokens=800, dola_layers=candidate_premature_layers, trust_remote_code=True)
	base_output = model.generate(**inputs, do_sample=False,max_new_tokens=800)
	df.loc[i,"FinalGeneratedStory"]=(tokenizer.batch_decode(dola_high_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))[0]
	df2.loc[i,"FinalGeneratedStory"]=(tokenizer.batch_decode(base_output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True))[0]
df.to_csv("/home/users/ntu/b230051/halcre/CS4/Generations/DoLa_Ins_Mistral7B_1.csv")
df2.to_csv("/home/users/ntu/b230051/halcre/CS4/Generations/Base_Ins_Mistral7B_1.csv")
