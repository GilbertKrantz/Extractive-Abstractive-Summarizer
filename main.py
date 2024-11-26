import Abstractive_Summarization.abstractive_summarizer as abstractive_summarizer
import Extractive_Summarization.extractive_summarizer as extractive_summarizer
import streamlit as st

def main():
    # Get user text input
    text = st.text_area("Enter the text you want to summarize: ")
    # Get user choice of summarization method
    summarization_method = st.selectbox("Choose the summarization method", ("Abstractive", "Extractive"))
    
    if st.button("Summarize"):
        if summarization_method == "Abstractive":
            # Summarize the text using abstractive summarization
            max_length = text.count(' ') + 1
            text_handler = abstractive_summarizer.textHandler(max_length=max_length)
            summary = text_handler.summarize(text)
            st.write(summary)
        elif summarization_method == "Extractive":
            # Summarize the text using extractive summarization
            text_handler = extractive_summarizer.textHandler()
            summary = text_handler.summarize(text)
            st.write(summary)
        
if __name__ == "__main__":
    main()