import streamlit as st
from transformers import pipeline
import yaml

# Load the YAML file
with open('cfg/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
PIPELINE_NAME = config["Config"]["PIPELINE_NAME"]
FINETUNED_MODEL_NAME = config["Config"]["FINETUNED_MODEL_NAME"]
pipe = pipeline(PIPELINE_NAME, model = FINETUNED_MODEL_NAME)

@st.cache
def extract_answer(question, context):
  return pipe(question=question,
        context=context)

def render():
    col1, col2 = st.columns(2)
    with col1:
        with st.form("form1",clear_on_submit=False):
            context = st.text_area("Enter your context here")
            question = st.text_input("Enter your question here")

            submit = st.form_submit_button("Submit",type="primary")
            
            if submit:
                with col2:
                    st.success('Done!')
                    result = extract_answer(question, context)
                    # Extract answer and score
                    answer = result['answer']
                    score = result['score']
                    # Display the answer in a box
                    st.subheader("Answer:")
                    st.info(f"🤖 {answer}")

                    # Display the score in a separate box
                    st.subheader("Score:")
                    st.info(score)