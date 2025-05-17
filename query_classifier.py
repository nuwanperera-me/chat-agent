from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryClassifier:
    """Classifies queries to determine the appropriate processing chain."""
    
    def __init__(self, llm):
        """Initialize the query classifier with a language model."""
        self.llm = llm
        self.parser = StrOutputParser()
        self._create_classifier_chain()
    
    def _create_classifier_chain(self):
        """Create the classification chain."""
        template = """
        You are a query classifier that determines the most appropriate system to handle a user query.
        
        Classify the following query into one of these categories:
        1. DOCUMENT - If the query is asking about information that might be found in documents or requires retrieving specific content from a knowledge base.
        2. TOOL - If the query is about weather, requires Wikipedia searches, needs Python code execution, or other tool-based operations.
        
        Query: {query}
        
        Classification (respond with only DOCUMENT or TOOL):
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        self.chain = prompt | self.llm | self.parser
    
    def classify(self, query):
        """Classify a query as either document-related or tool-related.
        
        Args:
            query (str): The user query to classify
            
        Returns:
            str: Either 'DOCUMENT' or 'TOOL'
        """
        result = self.chain.invoke({"query": query})
        # Clean and normalize the result
        result = result.strip().upper()
        
        # Ensure we only return valid classifications
        if result not in ["DOCUMENT", "TOOL"]:
            # Default to TOOL if classification is unclear
            return "TOOL"
            
        return result