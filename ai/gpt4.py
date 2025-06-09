import requests
from ai.base_processor import BaseAIProcessor
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage

class GPT4Processor(BaseAIProcessor):
    """Handles processing text using OpenAI's GPT-4 through LangChain"""
    
    def __init__(self, api_key, model="gpt-4-turbo-preview"):
        super().__init__("GPT-4 Turbo")
        self.api_key = api_key
        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=0,
            model_kwargs={"response_format": {"type": "text"}}
        )
    
    def _execute_model_request(self, text):
        """Send text to OpenAI's GPT-4-Turbo for MCQ analysis using LangChain"""
        print("Processing text with GPT-4-Turbo...")
        
        # Create messages for the chat
        system_message = SystemMessage(
            content="You are an expert at solving multiple choice questions. Analyze the given question and options, then provide the most likely correct answer with a brief explanation. Be concise and direct."
        )
        human_message = HumanMessage(content=text)
        
        # Create chat template
        chat_template = ChatPromptTemplate.from_messages([
            system_message,
            human_message
        ])
        
        # Get response from the model
        try:
            response = self.llm.invoke(chat_template.format_messages())
            return response.content
        except Exception as e:
            print(f"Error processing with GPT-4: {str(e)}")
            return None