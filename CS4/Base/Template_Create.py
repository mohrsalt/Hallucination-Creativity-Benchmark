from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import pandas as pd
import numpy as np
import torch
from accelerate.test_utils.testing import get_backend
from transformers import BitsAndBytesConfig
df=pd.read_csv("/home/users/ntu/b230051/halcre/CS4/Dataset/Base_Ins_Template.csv")
df2=pd.read_csv("/home/users/ntu/b230051/halcre/CS4/Dataset/Base_Story_Template.csv")


df["FinalGeneratedStory"] = ""
df2["FinalGeneratedStory"] = ""

df.to_csv("/home/users/ntu/b230051/halcre/CS4/Dataset/Base_Ins_Template_Clean.csv")
df2.to_csv("/home/users/ntu/b230051/halcre/CS4/Dataset/Base_Story_Template_Clean.csv")
