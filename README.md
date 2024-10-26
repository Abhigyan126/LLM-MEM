# LLM-MEM 🧠
### An Enhanced Language Model with Semantic Memory Storage

A Python-based implementation that augments Language Models with persistent semantic memory using vector embeddings and similarity search.

## 🌟 Features

- Semantic chunking of conversations using NLTK
- Vector embeddings generation using Sentence Transformers
- Cosine similarity-based context retrieval
- Persistent memory storage using vector database
- Automatic context injection for more coherent responses
- Smart text chunking with code block preservation

## 🛠️ Technical Stack

- `sentence-transformers`: For generating semantic embeddings
- `nltk`: Natural Language Processing toolkit for text manipulation
- Custom LLM integration
- [Vector-store](https://github.com/Abhigyan126/Vector-Store) for storing and retrieving embeddings
- REST API endpoints for vector operations

## 📋 Prerequisites

- sentence-transformers
- nltk
- requests
- google-generativeai


## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/Abhigyan126/LLM-MEM.git
cd LLM-MEM
```

2. Install dependencies:
```bash
pip install sentence-transformers nltk requests google-generativeai
```
3. create .env file
```bash
   key = your_llm_key
   ```
4. Run the main script:
```bash
python main.py
```

## 💡 How It Works

1. **Text Processing**: Incoming messages are processed and cleaned
2. **Embedding Generation**: Converts text into vector embeddings
3. **Semantic Search**: Finds relevant previous contexts using cosine similarity
4. **Context Integration**: Merges relevant history with current query
5. **Response Generation**: Generates response using LLM with enhanced context
6. **Memory Storage**: Stores new interactions for future reference

## 🔧 Configuration

The system uses several configurable parameters:
- Maximum chunk length: 2000 characters
- Minimum chunk size: 100 characters
- Similarity threshold: 0.5
- Default nearest neighbors: 10
