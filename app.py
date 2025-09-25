import os
import streamlit as st
import time
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ------------------ Load environment ------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="AskMate - PDF Q&A", page_icon="📄", layout="wide")
st.title("📄 AskMate - Ask Questions from Your PDFs")
st.write("Upload PDF(s) and ask questions. Powered by Google Gemini LLM.")

#  Sidebar
st.sidebar.title("⚙️ Options")
clear_chat = st.sidebar.button("🗑️ Clear Chat History")
download_chat = st.sidebar.button("⬇️ Download Chat History")


with st.sidebar.expander("ℹ️ About this App"):
    st.write("This app lets you ask questions from your uploaded PDFs. It uses:")
    st.markdown("- 🧠 **FAISS** for vector search\n- ✨ **Gemini LLM** for answers\n- 📑 **RAG** pipeline")


# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if clear_chat:
    st.session_state.chat_history = []
    st.rerun()
    # ensures UI is refreshed and old chats disappear
    st.success("Chat history cleared!")

# ------------------ File Upload ------------------
uploaded_files = st.file_uploader("Upload your PDF(s)", type="pdf", accept_multiple_files=True)

docs_texts = []
if uploaded_files:
    os.makedirs("temp", exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner(f"📂 Processing {uploaded_file.name}..."):
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            docs_texts.extend(pages)
            time.sleep(1)  # just for smoother UX
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")

# ------------------ Create VectorStore ------------------
qa_chain = None
if docs_texts:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs_texts)

    if documents:  # ✅ Ensure documents is not empty
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(documents, embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        # 🔑 Custom prompt: allows LLM to fallback on its own knowledge
        custom_prompt = PromptTemplate(
            template="""
You are AskMate, a helpful assistant. 
Use the following context from documents if relevant. 
If the context does not contain the answer, use your own knowledge to answer.

Context:
{context}

Question:
{question}

Answer as clearly and helpfully as possible.
""",
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=True
        )
    else:
        st.error("❌ Could not extract any text from the uploaded PDF(s). Please try another file.")

# ------------------ Ask Questions ------------------
user_question = st.text_input("💬 Ask your question here:")

if user_question and qa_chain:
    with st.spinner("Thinking..."):
        try:
            result = qa_chain({"query": user_question})
            answer = result["result"]
            sources = result.get("source_documents", [])

            # Save chat history
            st.session_state.chat_history.append(("You", user_question))
            st.session_state.chat_history.append(("AskMate", answer))

            # Show Answer
            st.markdown("### 🤖 Answer:")
            st.write(answer)

            # Show Sources
            if sources:
                with st.expander("📑 Sources"):
                    for src in sources:
                        st.write(f"- Page {src.metadata.get('page', 'N/A')} | {src.metadata.get('source', '')}")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

# ------------------ Show Chat History ------------------
if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    for role, message in st.session_state.chat_history:
        st.write(f"**{role}:** {message}")

# ------------------ Download Option ------------------
if download_chat and st.session_state.chat_history:
    chat_text = "\n".join([f"{role}: {msg}" for role, msg in st.session_state.chat_history])
    st.download_button("Download Chat History", chat_text, file_name="askmate_chat.txt")
