import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Pandas Agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# RAG
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# LLM
from langchain_groq import ChatGroq


# ------------------------
# Load API Key
# ------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key or not api_key.startswith("gsk_"):
    st.error("🚨 API Key Error: Add GROQ_API_KEY inside .env file")
    st.stop()


# ------------------------
# 1. Process Data
# ------------------------
def process_data(file_path):

    df = pd.read_csv(file_path)

    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Ship_Date'] = pd.to_datetime(df['Ship_Date'])

    df['Shipping_Delay'] = (df['Ship_Date'] - df['Order_Date']).dt.days
    df['Order_Processing_Time'] = 1

    temp_path = "temp_processed_orders.csv"
    df.to_csv(temp_path, index=False)

    return df, temp_path


# ------------------------
# 2. Setup RAG
# ------------------------
@st.cache_resource
def setup_rag_system(csv_path):

    loader = CSVLoader(file_path=csv_path)
    documents = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(documents, embeddings)

    retriever = vector_db.as_retriever(search_kwargs={"k":5})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=api_key
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    return qa_chain


# ------------------------
# 3. Router
# ------------------------
def route_question(query, llm):

    prompt = f"""
You are a classification router.

If the question requires calculations, averages,
counting, or math operations → reply MATH.

If the question asks about specific records,
details, or document retrieval → reply SEARCH.

Question: {query}

Reply with only one word.
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().upper()

    if "MATH" in decision:
        return "MATH"

    return "SEARCH"


# ------------------------
# STREAMLIT UI
# ------------------------
st.set_page_config(page_title="Supply Chain Copilot", layout="wide")

st.title("🚀 Supply Chain AI Copilot")

st.write("Agentic Routing + RAG + Data Analysis")

uploaded_file = st.file_uploader("Upload CSV", type="csv")


if uploaded_file is not None:

    df, temp_csv_path = process_data(uploaded_file)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=api_key
    )

    pandas_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )

    rag_chain = setup_rag_system(temp_csv_path)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Data Preview")
        st.dataframe(df.head())

    with col2:
        st.write("### Average Shipping Delay")

        if 'Warehouse' in df.columns:
            st.bar_chart(
                df.groupby('Warehouse')['Shipping_Delay'].mean(),
                width="stretch"
            )

    st.divider()

    st.write("### 🤖 Ask the AI Copilot")

    user_question = st.text_input("Ask a supply chain question")

    if user_question:

        with st.spinner("Thinking..."):

            route = route_question(user_question, llm)

            st.caption(f"Routed to: {route}")

            if route == "MATH":

                result = pandas_agent.invoke(
                    {"input": user_question}
                )

                answer = result["output"]

            else:

                answer = rag_chain.run(user_question)

            st.success(answer)