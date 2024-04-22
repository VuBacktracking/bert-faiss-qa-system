import torch
from transformers import AutoTokenizer, AutoModel
import yaml

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the YAML file
with open('cfg/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
DATASET_NAME = config["Config"]["DATSET_NAME"]
MODEL_NAME = config["Config"]["MODEL_NAME"]
EMBEDDING_COLUMN = config["Config"]["EMBEDDING_COLUMN"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

def cls_pooling(model_output):
  return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
  encoded_input = tokenizer(
      text_list,
      padding = True,
      truncation = True,
      return_tensors = "pt"
  )
  encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
  model_output = model(**encoded_input)

  return cls_pooling(model_output)