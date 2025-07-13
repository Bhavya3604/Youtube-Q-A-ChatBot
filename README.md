# Youtube-Q&A-ChatBot

A Retrieval-Augmented Generation (RAG) based chatbot that allows users to ask questions about YouTube videos by leveraging their transcripts. Built using LangChain, OpenAI, FAISS, and Streamlit.

# Features

- Automatically retrieves YouTube video transcripts using `youtube-transcript-api`
- Embeds and indexes transcript chunks with OpenAI's embedding model and FAISS
- Generates context-aware answers using GPT-4o-mini from OpenAI
- Streamlit-based UI for interactive querying
- Implements a full RAG pipeline using LangChain

# Tech Stack

- LangChain
- OpenAI API (Embeddings & GPT-4o-mini)
- FAISS (Vector Store)
- Streamlit (Frontend)
- youtube-transcript-api
- Python 3.9+




