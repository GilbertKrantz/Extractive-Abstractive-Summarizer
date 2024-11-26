import Abstractive_Summarization.abstractive_summarizer as abstractive_summarizer
import Extractive_Summarization.extractive_summarizer as extractive_summarizer
import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm
import matplotlib.pyplot as plt

class loadTest:
    def __init__(self, file_path, separator, encoding) -> None:
        self.data = pd.read_csv(file_path, sep=separator, encoding=encoding)
        self.data_len = len(self.data)
    
    def get_data(self):
        return self.data
    
    def get_text(self, index):
        return self.data.iloc[index]['text']
    
    def get_summary(self, index):
        return self.data.iloc[index]['summary']

class testModels:
    def __init__(self) -> None:
        self.abstractive_summarizer = abstractive_summarizer.textHandler()
        self.extractive_summarizer = extractive_summarizer.textHandler()
        self.loadTest = loadTest()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    def test_abstractive_summarizer(self, index):
        text = self.loadTest.get_text(index)
        summary = self.loadTest.get_summary(index)
        predicted_summary = self.abstractive_summarizer.summarize(text)
        scores = self.scorer.score(summary, predicted_summary)
        return scores
    
    def test_extractive_summarizer(self, index):
        text = self.loadTest.get_text(index)
        summary = self.loadTest.get_summary(index)
        predicted_summary = self.extractive_summarizer.summarize(text)
        scores = self.scorer.score(summary, predicted_summary)
        return scores
    
    def test_all(self, max_index=10):
        results = []
        # Use tqdm to wrap the range and show progress
        for i in tqdm(range(max_index), desc="Testing Summarization Models", unit="document"):
            abstractive_scores = self.test_abstractive_summarizer(i)
            extractive_scores = self.test_extractive_summarizer(i)
            results.append({'abstractive': abstractive_scores, 'extractive': extractive_scores})
        return results
    
    def plot_results(self, results):
        # Plot the results using matplotlib
        rouge1_abstractive = [result['abstractive']['rouge1'].fmeasure for result in results]
        rouge1_extractive = [result['extractive']['rouge1'].fmeasure for result in results]
        rougeL_abstractive = [result['abstractive']['rougeL'].fmeasure for result in results]
        rougeL_extractive = [result['extractive']['rougeL'].fmeasure for result in results]
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(rouge1_abstractive, label='Abstractive')
        ax[0].plot(rouge1_extractive, label='Extractive')
        ax[0].set_title('Rouge-1 F1 Score')
        ax[0].legend()
        
        ax[1].plot(rougeL_abstractive, label='Abstractive')
        ax[1].plot(rougeL_extractive, label='Extractive')
        ax[1].set_title('Rouge-L F1 Score')
        ax[1].legend()
        
        plt.show()
        
        return fig
        
        

def main():
    tester = testModels()
    results = tester.test_all()
    print(results)

if __name__ == "__main__":
    main()