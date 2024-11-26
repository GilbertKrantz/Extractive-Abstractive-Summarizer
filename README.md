# Streamlit Abstractive and Extractive Text Summarizer

This project is a web application built with Streamlit that provides both abstractive and extractive text summarization. The abstractive summarization is powered by the Hugging Face Transformer model `t5-small`, while the extractive summarization uses the BERT Summarizer.

## Features

- **Abstractive Summarization**: Generates a summary by interpreting the text and generating new sentences.
- **Extractive Summarization**: Selects key sentences directly from the text to form a summary.
- **Model Testing**: Test the summarization models using a dataset and evaluate their performance.

## Models Used

- **Abstractive Summarization**: [Hugging Face Transformer `t5-small`](https://huggingface.co/t5-small)
- **Extractive Summarization**: [BERT Summarizer](https://github.com/dmmiller612/bert-extractive-summarizer)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/streamlit-text-summarizer.git
    cd streamlit-text-summarizer
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to use the application.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the `t5-small` model.
- [BERT Extractive Summarizer](https://github.com/dmmiller612/bert-extractive-summarizer) for the extractive summarization model.
## Live Demo

Check out the live demo of the application [here](https://extractive-abstractive-summarizer-dqi8mgzmkmjezuimdkycg6.streamlit.app/).