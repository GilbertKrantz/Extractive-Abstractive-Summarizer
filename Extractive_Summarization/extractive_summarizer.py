from summarizer import Summarizer

class ExtractiveSummarizer:
    def __init__(self):
        self.model = Summarizer()

    def summarize(self, text):
        return self.model(text)
    
class textHandler:
    def __init__(self):
        self.text = None
        
    def summarize(self, text):
        self.text = text
        summarizer = ExtractiveSummarizer()
        return summarizer.summarize(self.text)
    
    def get_text(self):
        return self.text
    
    def set_text(self, text):
        self.text = text
        
def main() :
    text = input("Enter the text you want to summarize: ")
    text_handler = textHandler()
    print(text_handler.summarize(text))
    
if __name__ == "__main__":
    main()