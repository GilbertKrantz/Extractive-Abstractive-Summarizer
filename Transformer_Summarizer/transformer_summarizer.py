from transformers import pipeline, AutoTokenizer

class TransformerSummarizer:
    def __init__(self, model_name="google-t5/t5-small"):
        self.summarizer = pipeline("summarization", model=model_name, device_map="auto")
        self.max_length = AutoTokenizer.from_pretrained(model_name).model_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def summarize(self, text):
        return self.summarizer(text, max_length=self.max_length)[0]['summary_text']
    
def main():
    text = input("Enter the text you want to summarize: ")
    summarizer = TransformerSummarizer()
    print(summarizer.summarize(text))
    
if __name__ == "__main__":
    main()
