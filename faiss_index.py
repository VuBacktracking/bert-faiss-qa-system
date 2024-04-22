import numpy as np
import collections
import torch
import faiss
import evaluate

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import yaml
from utils.embedding import get_embeddings

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the YAML file
with open('cfg/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

DATASET_NAME = config["Config"]["DATSET_NAME"]
MODEL_NAME = config["Config"]["MODEL_NAME"]
EMBEDDING_COLUMN = config["Config"]["EMBEDDING_COLUMN"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

raw_datasets = load_dataset(DATASET_NAME, split = "train+validation")
raw_datasets = raw_datasets.filter(
    lambda x: len(x['answers']['text']) > 0
)

if __name__ == "__main__":
    
    embeddings_dataset = raw_datasets.map(
        lambda x: {
            EMBEDDING_COLUMN: get_embeddings(x["question"]).detach().cpu().numpy()[0]
        }
    )
    embeddings_dataset.add_faiss_index(column=EMBEDDING_COLUMN)
    embeddings_dataset.save_faiss_index(EMBEDDING_COLUMN, "faiss_index/my_index.faiss")