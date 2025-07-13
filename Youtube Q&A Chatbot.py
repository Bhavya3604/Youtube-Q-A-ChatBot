# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")
st.title("🎥 YouTube Transcript Chatbot")
st.markdown("Ask questions based on a YouTube video's transcript.")

# Sidebar for video input
video_id = st.text_input("Enter YouTube Video ID (e.g., Gfr50f6ZBvo):")

if video_id:
    try:
        st.info("Fetching transcript...")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        st.success("Transcript retrieved successfully!")
        with st.expander("📜 Show Transcript"):
            st.write(transcript)
    except TranscriptsDisabled:
        st.error("❌ No captions available for this video.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

    # Step 1b - Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Step 1c - Embeddings & FAISS Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Prompt template
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    # Chain setup
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    main_chain = parallel_chain | prompt | llm | parser

    # Question input
    question = st.text_input("Ask a question about the video:")

    if st.button("Ask") and question.strip() != "":
        with st.spinner("Thinking..."):
            try:
                answer = main_chain.invoke(question)
                st.success("Answer:")
                st.markdown(f"**{answer}**")
            except Exception as e:
                st.error(f"Failed to answer: {str(e)}")
