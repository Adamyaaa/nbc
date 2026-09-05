from typing import Type, TypeVar, Optional, Any
from pydantic import BaseModel
from google import genai
from google.genai import types

from app.core.config import settings

T = TypeVar("T", bound=BaseModel)

class LLMClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=self.api_key)
        self.default_model = "gemini-2.5-flash" # Default to a fast, reliable model

    def generate_text(self, prompt: str, model: Optional[str] = None, system_instruction: Optional[str] = None) -> str:
        """Basic text generation."""
        model_name = model or self.default_model
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2, # Low temperature for more deterministic RAG answers
        )
        
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        return response.text

    def generate_structured(self, prompt: str, schema: Type[T], model: Optional[str] = None, system_instruction: Optional[str] = None) -> T:
        """Structured text generation returning a Pydantic model."""
        model_name = model or self.default_model
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=schema,
        )
        
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        
        # Pydantic validation
        return schema.model_validate_json(response.text)

    def generate_embedding(self, text: str, model: str = "text-embedding-004") -> list[float]:
        """Generate embedding for a given text."""
        result = self.client.models.embed_content(
            model=model,
            contents=text,
        )
        return result.embeddings[0].values
