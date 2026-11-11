
import streamlit as st
from transformers import pipeline

# Application title
st.title("BERT Question Answering System")

# Application description
st.write(
    "Enter a paragraph and ask a question. "
    "BERT will find the answer."
)

# Loading the pretrained BERT model
@st.cache_resource
def load_bert_model():
    qa_model = pipeline(
        "question-answering",
        model="deepset/bert-base-cased-squad2"
    )

    return qa_model

# Load the model
qa_model = load_bert_model()

st.success("BERT model loaded successfully!")

# Input paragraph
context = st.text_area(
    "Enter your paragraph:",
    height=200
)

# Input question
question = st.text_input(
    "Enter your question:"
)

# Answer button
if st.button("Find Answer"):

    if context.strip() == "" or question.strip() == "":
        st.warning("Please enter both paragraph and question.")

    else:
        # Pass question and context to BERT
        result = qa_model(
            question=question,
            context=context
        )

        # Display answer
        st.subheader("Answer")
        st.success(result["answer"])

        # Display confidence score
        st.write(
            "Confidence Score:",
            round(result["score"], 4)
        )
