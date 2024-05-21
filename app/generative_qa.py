import numpy as np
import streamlit as st
from transformers import pipeline
from datasets import load_dataset
import yaml
from faiss_index import get_embeddings


# Load the YAML file
with open('cfg/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
DATASET_NAME = "squad_v2"
EMBEDDING_COLUMN = config["Config"]["EMBEDDING_COLUMN"]
PIPELINE_NAME = config["Config"]["PIPELINE_NAME"]
FINETUNED_MODEL_NAME = config["Config"]["FINETUNED_MODEL_NAME"]
TOP_K = config["Config"]["TOP_K"]

pipe = pipeline(PIPELINE_NAME, model = FINETUNED_MODEL_NAME)
dataset = load_dataset(DATASET_NAME, split = "train+validation")
dataset = dataset.filter(
    lambda x: len(x['answers']['text']) > 0
)
dataset.load_faiss_index('question_embedding', 'faiss_index/my_index.faiss')

@st.cache
def generate_answer(input_question):
    input_question_embedding = get_embeddings([input_question])
    input_question_embedding = input_question_embedding.cpu().detach().numpy()
    scores, samples = dataset.get_nearest_examples(
    EMBEDDING_COLUMN, input_question_embedding, k = TOP_K
)
    answer_scores = []
    for idx, score in enumerate(scores):
        question = samples["question"][idx]
        context = samples["context"][idx]
        answer = pipe(
            question=question,
            context=context
        )
        answer_scores.append(answer["score"])

    best_index = np.argmax(answer_scores)
    _question = samples["question"][best_index]
    _context = samples["context"][best_index]

    _answer = pipe(
        question=_question,
        context=_context
    )    
    return _answer['answer'], _answer['score']

def render():
    with st.form("form2", clear_on_submit=False):
        question = st.text_area("Question:")
        
        submit = st.form_submit_button("Submit", type="primary")

        if submit:
            st.success('Done!')
            answer, score = generate_answer(question)
            # Display the answer in a box
            st.subheader("Answer:")
            st.info(f"🤖 {answer}")

            # Display the score in a separate box
            st.subheader("Score:")
            st.info(score)