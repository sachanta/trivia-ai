import os
from dotenv import load_dotenv
from google.cloud import vision
from google.oauth2 import service_account
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

class Config:
    """Configuration class for application settings and API credentials"""
    
    def __init__(self):
        load_dotenv()
        
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
        self.google_api_key = os.getenv("GEMINI_API_KEY")
        self.phoenix_api_key = os.getenv("PHOENIX_API_KEY")
        
        self._validate_credentials()
        self._init_phoenix_tracing()
        
        self.vision_client = self._init_vision_client()
    
    def _validate_credentials(self):
        """Validate that required credentials are set"""
        if not self.perplexity_api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable is not set or empty")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set or empty")
        if not self.google_credentials_path:
            raise ValueError("GOOGLE_CREDENTIALS_PATH environment variable is not set or empty")
        if not self.google_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set or empty")
        if not self.phoenix_api_key:
            raise ValueError("PHOENIX_API_KEY environment variable is not set or empty")
    
    def _init_vision_client(self):
        """Initialize and return a Google Vision client"""
        credentials = service_account.Credentials.from_service_account_file(
            self.google_credentials_path
        )
        return vision.ImageAnnotatorClient(credentials=credentials)
    
    def _init_phoenix_tracing(self):
        """Initialize Phoenix tracing and monitoring"""
        # Set Phoenix environment variables
        os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={self.phoenix_api_key}"
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"
        
        # Configure Phoenix tracer
        tracer_provider = register(
            project_name="trivia-gpt",  # Your project name
        )
        
        # Instrument LangChain
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)