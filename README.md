# MIMIC Medical Chatbot with RAG

This project implements a medical chatbot powered by a Retrieval-Augmented Generation (RAG) architecture. It leverages medical textbooks from the MIMIC-III dataset as its knowledge base.

## Project Description

The chatbot is designed to answer medical questions by retrieving relevant information from a curated collection of medical textbooks and then using a large language model to generate a coherent and informative response based on the retrieved context. This approach aims to improve the accuracy and relevance of the chatbot's answers compared to using a language model alone.

The project includes the following key components:

- **Data Loading and Preprocessing:** Downloading, extracting, cleaning, and chunking medical textbook data.
- **Embeddings Generation:** Using a pre-trained sentence transformer model to create vector representations of the text chunks.
- **FAISS Indexing:** Building a FAISS index for efficient similarity search over the text chunk embeddings.
- **Document Retrieval:** Retrieving the most relevant text chunks based on a user's query using the FAISS index.
- **Response Generation:** Utilizing a large language model (Phi-3.5-mini-instruct) to generate a natural language response based on the user's query and the retrieved context.
- **Flask Application:** A simple Flask web application to expose the chatbot's functionality via an API endpoint.

## Getting Started

To run this project, you will need a Google Colab environment or a similar environment with GPU support.

1. **Clone the repository:**
`git clone https://github.com/mohamed-essam1/MIMIC-Medical-Chatbot-RAG.git`

2. **Open the Colab notebook:**
   Upload the `MIMIC-Medical-Chatbot-RAG.ipynb` notebook to your Google Drive and open it in Google Colab.
3. **Run the cells:**
   Execute the notebook cells sequentially. This will install dependencies, download and process data, set up the FAISS index, start the Flask application, and provide functions for interacting with the chatbot.

## Dependencies

The project relies on the following libraries:

- `requests`
- `tqdm`
- `faiss-cpu`
- `transformers`
- `tensorflow`
- `sentence-transformers`
- `textblob`
- `gensim`
- `numba`
- `flask`
- `torch`
- `gc`
- `json`
- `os`
- `logging`

These dependencies are installed automatically when you run the first cell in the Colab notebook.

## Usage

After running all the cells in the Colab notebook, the Flask application will be running in the background. You can interact with the chatbot using the `ask_chatbot` function defined in the notebook.

python `print(ask_chatbot("Your medical query here"))`

Replace `"Your medical query here"` with the medical question you want to ask the chatbot.

## File Structure

- `MIMIC-Medical-Chatbot-RAG.ipynb`: The main Google Colab notebook containing all the code for the project.
- `app.py`: The Python script for the Flask application.
- `faiss_index.idx`: The saved FAISS index (generated after the first run).
- `chunked_documents.json`: The saved chunked medical textbook documents (generated after the first run).
- `mimic_textbooks/`: Directory for downloaded and extracted textbook data.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. (You may need to create a LICENSE file with the MIT license text).

## Acknowledgments

- The developers of the MIMIC-III dataset for providing valuable medical text data.
- The libraries used in this project (transformers, sentence-transformers, FAISS, Flask, etc.).
