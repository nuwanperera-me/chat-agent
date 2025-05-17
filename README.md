# AI Assistant with RAG System

This project implements an AI assistant that can answer questions about the weather, search Wikipedia for information, run Python code, and retrieve information from documents using a Retrieval Augmented Generation (RAG) system.

## Features

- Weather information retrieval
- Wikipedia search
- Python code execution
- Document retrieval using RAG
- Conversation history management

## Setup

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Add your documents to the `docs` directory (text files with .txt extension)

## Usage

Run the assistant:

```
python main.py
```

The assistant will initialize the RAG system and load documents from the `docs` directory. You can then interact with the assistant by typing your questions.

### Example Queries

- "What's the weather like in New York?" (requires latitude and longitude)
- "Tell me about artificial intelligence" (uses Wikipedia)
- "Calculate 15 * 24 + 7" (runs Python code)
- "What is RAG and how does it work?" (uses the RAG system to retrieve information from documents)

## RAG System

The RAG system enhances the assistant by retrieving relevant information from documents before generating responses. This improves accuracy and allows the assistant to access specific knowledge stored in the documents.

To add your own knowledge to the system, simply add text files to the `docs` directory. The system will automatically process and index these documents for retrieval.