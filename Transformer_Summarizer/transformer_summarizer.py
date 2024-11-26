from transformers import pipeline

class TransformerSummarizer:
    def __init__(self, model_name="google-t5/t5-small"):
        self.summarizer = pipeline("summarization", model=model_name, device=0)
    
    def summarize(self, text, max_length=150):
        return self.summarizer(text, max_length=max_length)[0]['summary_text']
    
def main():
    text = input("Enter the text you want to summarize: ")
    summarizer = TransformerSummarizer()
    print(summarizer.summarize(text))
    
if __name__ == "__main__":
    main()