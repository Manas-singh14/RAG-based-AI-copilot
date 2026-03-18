# 📦 Supply Chain AI Copilot

An intelligent, multi-agent AI assistant designed to help supply chain managers analyze order shipment data, compute delays, and answer complex operational questions using natural language. 

[cite_start]Built as a submission for the AI Engineering Intern Challenge[cite: 1, 2, 4].

## ✨ Features Implemented

[cite_start]This project fulfills all minimum requirements [cite: 14] [cite_start]and includes all 5 bonus features  to deliver a comprehensive dashboard:

- [cite_start]**Automated Data Processing:** Instantly computes `Order processing time` and `Shipping delay` (`Ship_Date - Order_Date`)[cite: 15, 16, 17, 18, 20].
- [cite_start]**Natural Language Question Answering:** Ask complex queries about warehouse performance, product speeds, and specific orders[cite: 21, 22].
- [cite_start]**Dual-Engine AI Architecture:** Utilizes a custom LLM router to dynamically switch between a Pandas Code Agent (for math/aggregations) and a FAISS Vector Database + RAG (for semantic search)[cite: 35].
- [cite_start]**Interactive Dashboard & Web Interface:** Built with Streamlit, featuring automated chart visualizations of warehouse delays[cite: 27, 32, 36].
- [cite_start]**Dynamic File Uploads:** Analyze any new supply chain CSV on the fly[cite: 33].
- [cite_start]**AI Explanations:** The system doesn't just return a number; it explains the operational logic and steps taken to calculate the result[cite: 34].

## 🧠 System Architecture Explanation

[cite_start]To handle both mathematical aggregations and specific text retrieval effectively, this Copilot uses a **Routed Multi-Agent System** powered by LangChain [cite: 43] and Groq (Llama-3.3-70b).

1. **The Intent Router:** When a user asks a question, an initial LLM call classifies the intent. 
2. [cite_start]**The Pandas DataFrame Agent (The "Math" Engine):** If the question requires averages, counting, or sorting (e.g., "What is the average delay per warehouse?" [cite: 12]), the query is routed to the Pandas Agent. The agent writes and executes temporary Python code to calculate the exact mathematical answer accurately.
3. [cite_start]**The Vector DB + RAG Pipeline (The "Search" Engine):** If the question asks for specific rows or contextual information (e.g., "Which orders were delayed more than 3 days?" [cite: 13]), the query is routed to the RAG pipeline. The CSV is chunked into documents, embedded using HuggingFace (`all-MiniLM-L6-v2`), and stored in a local FAISS index for semantic retrieval.

[cite_start]*Note on LLM Selection: While the assignment listed OpenAI/Claude APIs[cite: 42], I opted to use the Groq API (running Meta's Llama 3.3 70B model) to demonstrate framework modularity via LangChain, avoid rate-limit bottlenecks during development, and achieve ultra-fast inference speeds. The code can be instantly reverted to OpenAI by simply swapping the `ChatGroq` class for `ChatOpenAI`.*

## ⚠️ System Limitations

**Limitation: Scalability with Large Datasets (Token Limits & Execution Risks)**
While the current architecture is highly accurate for standard CSV files, the **Pandas DataFrame Agent** struggles with massive, enterprise-scale datasets (e.g., millions of rows). Because the agent needs to inject the dataframe's schema (and sometimes row samples) into the LLM's context window, very large files can exceed token limits or increase latency. Furthermore, allowing an LLM to generate and execute Python code (`allow_dangerous_code=True`) poses a security risk in a live production environment. A more robust enterprise solution would execute the generated code within an isolated Docker sandbox.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd supply_chain_copilot

2. **Create a virtual environment (Optional but recommended):**
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install dependencies:**
pip install pandas streamlit langchain langchain-experimental langchain-groq langchain-classic langchain-community langchain-huggingface faiss-cpu sentence-transformers python-dotenv

4. **Set up Environment Variables:**
Create a .env file in the root directory and add your Groq API key:
GROQ_API_KEY=your_api_key_here

5. **Run the Application:**
python -m streamlit run ultimate_app.py
