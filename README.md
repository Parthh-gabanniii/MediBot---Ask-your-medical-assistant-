🏥 Medie Bot – AI-Powered Medical Chatbot
Medie Bot is an end-to-end AI chatbot built specifically for medical and healthcare use cases. It leverages LangChain, Hugging Face models (Mistral), semantic search with vector embeddings, and a Streamlit UI, ensuring trustworthy answers sourced only from verified medical textbooks.

🚀 Project Highlights
✅ Trusted Medical Knowledge: Uses chunked data from vetted medical textbooks (PDFs).
🧠 LLM Integration: Employs Mistral via Hugging Face APIs for accurate, domain-specific answers.
🔍 Semantic Search: Embeddings + FAISS enable precise context retrieval under token limits.
💬 Conversational UI: Streamlit-based app shows chat history, markdown answers, and sources.
🔐 Zero Hallucination Policy: Responses restricted to known data; refuses to guess.
📦 Open Source & Scalable: Designed for developer portfolios, freelance work, and further expansion.


⚙️ Tech Stack
Component	Tool/Framework
UI	Streamlit
LLM Backend	Mistral via Hugging Face
Embeddings	Sentence Transformers
Vector Store	FAISS
AI Orchestration	LangChain
Data Source	Curated Medical Textbooks (PDFs)



📚 Architecture Overview
Phase 1: Memory & Data Ingestion
Extracts and chunks text from trusted medical PDFs.
Converts text chunks into vector embeddings using sentence transformers.
Stores embeddings and metadata in a FAISS vector DB.

Phase 2: LLM Integration & Query Handling
User queries are embedded and matched semantically against the vector DB.
Contextual chunks are passed to the Mistral model via LangChain for response generation.
Responses are strictly limited to the retrieved medical context.


 Future Improvements
🔐 User authentication for secure access.
📤 PDF upload to customize knowledge base.
🧪 Unit testing & modular LLM chaining.
⚙️ Scalability and cloud deployment options.

