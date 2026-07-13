# ==============================================
# STEP 1: IMPORTS AUR SETUP
# ==============================================
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
import os

# OpenAI API Key set karein (aapki apni key use karein)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"


# STEP 2: SAMPLE DOCUMENTS BANAYEIN (for demo)

documents = [
    {
        "page_content": "Machine Learning is a subset of AI that uses algorithms to learn from data.",
        "metadata": {"category": "AI", "source": "ml_doc1"}
    },
    {
        "page_content": "HR policies include employee benefits, leave management, and performance reviews.",
        "metadata": {"category": "HR", "source": "hr_doc1"}
    },
    {
        "page_content": "Deep Learning uses neural networks with multiple layers to process complex data.",
        "metadata": {"category": "AI", "source": "ml_doc2"}
    },
    {
        "page_content": "Employee onboarding process includes documentation, training, and orientation.",
        "metadata": {"category": "HR", "source": "hr_doc2"}
    },
    {
        "page_content": "Supervised learning requires labeled training data for model training.",
        "metadata": {"category": "AI", "source": "ml_doc3"}
    }
]

# STEP 3: VECTORSTORE CREATE KAREIN

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=[doc["page_content"] for doc in documents],
    embedding=embeddings,
    metadatas=[doc["metadata"] for doc in documents]
)

# STEP 4: HYBRID SEARCH (Similarity Search)

print("="*50)
print("1. HYBRID SEARCH (Similarity Search)")
print("="*50)

hybrid_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Query karein
query = "What is Machine Learning?"
results = hybrid_retriever.invoke(query)

print(f"\nQuery: {query}")
print("\nRetrieved Documents (Top 3):")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content[:100]}... (Category: {doc.metadata.get('category', 'N/A')})")

# STEP 5: METADATA FILTERING

print("\n" + "="*50)
print("2. METADATA FILTERING (HR Category)")
print("="*50)

filtered_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {
            "category": "HR"
        }
    }
)

# Query karein
filtered_results = filtered_retriever.invoke("employee policies")

print(f"\nQuery: employee policies")
print("\nRetrieved Documents (Only HR category):")
for i, doc in enumerate(filtered_results, 1):
    print(f"{i}. {doc.page_content} (Category: {doc.metadata.get('category')})")

# STEP 6: RAG CHAIN BANAYEIN AUR EVALUATION

print("\n" + "="*50)
print("3. RAG EVALUATION")
print("="*50)

# LLM initialize karein
llm = OpenAI(temperature=0)

# RAG chain banayein with metadata filtering
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=filtered_retriever  # HR category ke documents use honge
)

# Question poochhein aur answer dekhain
question = "What are the HR policies?"
answer = rag_chain.invoke(question)

print(f"\nQuestion: {question}")
print(f"Answer: {answer}")

# STEP 7: RAG EVALUATION METRICS (Manual)

print("\n" + "="*50)
print("4. RAG EVALUATION METRICS")
print("="*50)

# Evaluation metrics manually check karein
context_docs = filtered_retriever.invoke(question)
print(f"\nEvaluation Metrics:")
print(f"- Number of retrieved documents: {len(context_docs)}")
print(f"- Retrieved document categories: {[doc.metadata.get('category') for doc in context_docs]}")
print(f"- Answer relevance: Check if answer addresses the question")
print(f"- Answer: {answer[:150]}...")

print("\n" + "="*50)
print(" RAG System Evaluation Complete!")
print("="*50)
