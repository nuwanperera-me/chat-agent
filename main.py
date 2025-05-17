import os
import openai

from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


from tools import get_weather, get_wikipedia, run_python_code
from rag_system import RAGSystem
from query_classifier import QueryClassifier

_ = load_dotenv(find_dotenv()) 
openai.api_key = os.getenv('OPENAI_API_KEY')

tools = [
    get_weather,
    get_wikipedia,
    run_python_code
]

model = init_chat_model("gpt-4", model_provider="openai")

system_message_content = """You are a helpful assistant who can answer questions about the weather, 
search Wikipedia for information, run Python code, and retrieve information from documents. 
Use the provided tools when appropriate to answer user questions accurately.

If you don't know the answer, just say that you don't know, don't try to make up an answer.
If user asks about the weather, don't give just numbers. Give some propper summery with some your suggestions for the weather condition.

When asked about the weather, ask for the location's latitude and longitude.
When asked to search for information, use the Wikipedia tool.
When asked to perform calculations or run code, use the Python code runner tool.
When asked about information that might be in documents, use the RAG system to retrieve relevant information."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message_content),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

chat_message_history = ChatMessageHistory()

def limit_message_history(chat_history, k=5):
    """Limit the chat history to the last k messages."""
    if len(chat_history.messages) > k * 2: 
        chat_history.messages = chat_history.messages[-(k * 2):]

agent = create_openai_functions_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools,
    verbose=True,
)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: chat_message_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Initialize RAG system
rag_system = RAGSystem()
rag_chain = None

# Initialize query classifier
query_classifier = None

# Initialize router chain
router_chain = None

def setup_chains():
    """Set up all the chains needed for the system"""
    global rag_chain, query_classifier, router_chain
    
    # Initialize RAG chain if not already done
    if rag_chain is None:
        rag_chain = rag_system.create_rag_chain(model)
    
    # Initialize query classifier if not already done
    if query_classifier is None:
        query_classifier = QueryClassifier(model)

def process_with_router(input_text):
    """Process input using the router to select the appropriate chain"""
    global rag_chain, query_classifier
    
    # Make sure chains are set up
    setup_chains()
    
    # If RAG is not initialized properly, fall back to regular agent
    if rag_chain is None:
        return None, "TOOL"
    
    # Classify the query
    query_type = query_classifier.classify(input_text)
    
    # Route to appropriate chain based on classification
    if query_type == "DOCUMENT":
        try:
            rag_answer = rag_system.query(rag_chain, input_text)
            return rag_answer, "DOCUMENT"
        except Exception as e:
            print(f"RAG error: {e}")
            return None, "TOOL"
    else:
        return None, "TOOL"

def chat_loop():
    print("Welcome to the AI Assistant! Type 'exit' to end the conversation.")
    print("This assistant can help with weather information, Wikipedia searches, running Python code, and retrieving information from documents.")
    print("Initializing systems...")
    rag_system.initialize()
    setup_chains()
    print("All systems initialized. You can now ask questions about documents in the 'docs' directory or use tools.")
    
    message_count = 0
    session_id = "default_session"
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nAI: Goodbye! Have a great day!")
            break
        
        message_count += 1
        if message_count > 5:
            limit_message_history(chat_message_history, k=5)
        
        # Use router to determine the appropriate chain
        answer, chain_type = process_with_router(user_input)
        
        if chain_type == "DOCUMENT" and answer and "not initialized" not in answer:
            print(f"\nAI (Document): {answer}")
        else:
            # Use agent for tool-based queries or if RAG failed
            result = agent_with_chat_history.invoke(
                {"input": user_input},
                {"configurable": {"session_id": session_id}}
            )
            
            print(f"\nAI (Tool): {result['output']}")

if __name__ == "__main__":
    chat_loop()



