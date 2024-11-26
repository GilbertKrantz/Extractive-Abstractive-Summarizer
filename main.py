import Abstractive_Summarization.abstractive_summarizer as abstractive_summarizer
import Extractive_Summarization.extractive_summarizer as extractive_summarizer
import test as test
import streamlit as st

def main():
    # Test the summarization models or summarize user input
    st.title("Text Summarization App")
    st.text("This app summarizes text using abstractive or extractive summarization")
    
    # Choose between testing the models or summarizing user input
    choice = st.radio("Choose an option", ("Test Summarization Models", "Summarize Text"))
    
    if choice == "Test Summarization Models":
        # input dataset for testing with streamlit
        st.text("Testing the summarization models")
        # Streamlit data input
        st.text('Make sure the dataset has a "text" and "summary" column')
        data = st.file_uploader("Upload a dataset", type=["csv"])
        
        st.text("Choose the maximum number of documents to test")
        max_index = st.number_input("Maximum number of documents", min_value=1, value=20)
        
        # test button
        if st.button("Test"):
            if data is not None and max_index is not None:
                # Load the test dataset
                test_models = test.testModels.loadTest(data, separator=",", encoding="latin")
                # Test the summarization models
                results = test_models.test_all(max_index=max_index)
                # Plot the results (Show the Rouge-1 and Rouge-L F1 scores)
                # Show the results in streamlit
                fig = test_models.plot_results(results)
                st.pyplot(fig)
                
                # Download the results as a CSV file
                if st.button("Download Results"):
                    st.write("Downloading results")
                    st.write(results)
                    st.write("Download complete")
            else:
                st.text("Please upload a dataset and choose the maximum number of documents to test")
    else:
            # Get user text input
            text = st.text_area("Enter the text you want to summarize: ")
            # Get user choice of summarization method
            if text != "":
                summarization_method = st.selectbox("Choose the summarization method", ("Abstractive", "Extractive"))
            else:
                st.text("Please enter some text to summarize")
            
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