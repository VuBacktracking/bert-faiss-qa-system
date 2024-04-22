import streamlit as st
import app.sidebar as sidebar
import app.extractive_qa as extractive_qa
import app.generative_qa as generative_qa
st.set_page_config(layout="wide")

page = sidebar.show()

hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("Q&A System")
st.caption("Q&A system made from DistilBERT and Faiss Vector Database")

footer="""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

if page=="Extractive Q&A":
    extractive_qa.render()
if page=="Closed Generative Q&A":
    generative_qa.render()