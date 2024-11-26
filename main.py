import Abstractive_Summarization.abstractive_summarizer as abstractive_summarizer
import Extractive_Summarization.extractive_summarizer as extractive_summarizer
import Transformer_Summarizer.transformer_summarizer as transformer_summarizer
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
                test_models = test.testModels(data=data, separator=",", encoding="utf-8")
                # Test the summarization models
                results = test_models.test_all(max_index=max_index)
                # Plot the results (Show the Rouge-1 and Rouge-L F1 scores)
                # Show the results in streamlit
                fig = test_models.plot_results(results)
                st.pyplot(fig)
                
                # Save the results to a CSV file
                resultsDF = test_models.save_toDF(results)
                st.download_button(label="Download Results", data=resultsDF.to_csv().encode('utf-8'), file_name="results.csv", mime="text/csv")
    else:
            # Get user text input
            text = st.text_area("Enter the text you want to summarize: ")
            # Get user choice of summarization method
            if text != "":
                summarization_method = st.selectbox("Choose the summarization method", ("Abstractive", "Extractive", "Transformer"))
            else:
                st.text("Please enter some text to summarize")
            
            if text != "":
                if summarization_method == "Transformer":
                    model = st.selectbox("Choose the transformer model", ("google-t5/t5-small"))
                    # if model == "others":
                    #     st.warning("MAY NOT WORK FOR ALL MODELS")
                    #     model = st.text_input("Enter the model name (from Hugging Face Transformers)")
            else:
                model = None
            
            if text != '':
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
                    elif summarization_method == "Transformer":
                        if model is not None:
                            # Check model context size
                            # Summarize the text using transformer summarization
                            summarizer = transformer_summarizer.TransformerSummarizer(model_name=model)
                            if summarizer.max_length < len(summarizer.tokenizer(text).input_ids):
                                st.warning("The text is too long for the model. Please enter a shorter text.")
                            elif summarizer.max_length > len(summarizer.tokenizer(text).input_ids):
                                # Padding the text to the maximum length
                                padding = summarizer.max_length - len(summarizer.tokenizer(text).input_ids)
                                text = text + " ".join(["" for i in range(padding)])
                                summary = summarizer.summarize(text)
                                st.write(summary)
        
if __name__ == "__main__":
    main()
