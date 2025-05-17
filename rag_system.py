import os

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGSystem:
    def __init__(self, docs_dir="./docs", embedding_model="text-embedding-ada-002"):
        self.docs_dir = docs_dir
        self.embedding_model = embedding_model
        self.vector_store = None
        self.retriever = None
        
        # Create docs directory if it doesn't exist
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
            with open(f"{docs_dir}/sample.txt", "w") as f:
                f.write("This is a sample document for the RAG system. Add your own documents to this directory.")
    
    def load_documents(self):
        """Load documents from the docs directory"""
        try:
            loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []
    
    def process_documents(self, documents):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def create_vector_store(self, chunks):
        """Create a vector store from document chunks"""
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        vector_store = FAISS.from_documents(chunks, embeddings)
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        return vector_store
    
    def initialize(self):
        """Initialize the RAG system"""
        documents = self.load_documents()
        if documents:
            chunks = self.process_documents(documents)
            self.create_vector_store(chunks)
            return True
        return False
    
    def get_retriever(self):
        """Get the retriever for the RAG system"""
        if not self.retriever:
            success = self.initialize()
            if not success:
                return None
        return self.retriever
    
    def create_rag_chain(self, llm):
        """Create a RAG chain with the given LLM"""
        retriever = self.get_retriever()
        if not retriever:
            return None
        
        # Create the RAG prompt
        template = """Answer the following question based on the provided context and your knowledge.
        
Context:
{context}

Question:
{input}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
    
    def query(self, retrieval_chain, query):
        """Query the RAG system"""
        if not retrieval_chain:
            return "RAG system not initialized properly. Please check if documents are loaded."
        
        result = retrieval_chain.invoke({"input": query})
        return result["answer"]