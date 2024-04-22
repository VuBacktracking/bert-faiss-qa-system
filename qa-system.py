from faiss_index import generate_vector_database, get_embeddings
from transformers import pipeline
from datasets import load_dataset
import yaml

# Load the YAML file
with open('cfg/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

DATASET_NAME = config["Config"]["DATSET_NAME"]
EMBEDDING_COLUMN = config["Config"]["EMBEDDING_COLUMN"]
PIPELINE_NAME = config["Config"]["PIPELINE_NAME"]
FINETUNED_MODEL_NAME = config["Config"]["FINETUNED_MODEL_NAME"]
TOP_K = 5

# Load dataset and Faiss index
dataset = load_dataset(DATASET_NAME, split = "train+validation")
dataset = dataset.filter(
    lambda x: len(x['answers']['text']) > 0
)
dataset.load_faiss_index(EMBEDDING_COLUMN, "faiss_index/my_index.faiss")

# Pipeline
pipe = pipeline(PIPELINE_NAME, model = FINETUNED_MODEL_NAME)

# Input question
input_question = "When did Beyonce start becoming popular?"

input_question_embedding = get_embeddings([input_question])
input_question_embedding = input_question_embedding.cpu().detach().numpy()

scores, samples = dataset.get_nearest_examples(
    EMBEDDING_COLUMN, input_question_embedding, k = TOP_K
)

if __name__ == "__main__":
  print(f'Input question: {input_question}')
  for idx, score in enumerate(scores):
    question = samples["question"][idx]
    context = samples["context"][idx]
    answer = pipe(
                    question=question ,
                  context=context )
    print(f'Top {idx + 1}\tScore: {score}')
    print(f'Context: {context}')
    print(f'Answer: {answer}')
    print("*-----------------------------------*")