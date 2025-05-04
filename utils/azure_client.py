import os
import asyncio
from openai import AsyncAzureOpenAI
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class AzureOpenAIClient:
    """
    Client for interacting with Azure OpenAI GPT-4o model.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: str = "2023-12-01-preview"
    ):
        """
        Initialize the Azure OpenAI client.
        
        Args:
            api_key: Azure OpenAI API key (defaults to environment variable)
            endpoint: Azure OpenAI endpoint URL (defaults to environment variable)
            deployment_name: Azure deployment name for GPT-4o (defaults to environment variable)
            api_version: Azure OpenAI API version
        """
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.api_version = api_version
        
        if not all([self.api_key, self.endpoint, self.deployment_name]):
            missing = []
            if not self.api_key:
                missing.append("AZURE_OPENAI_API_KEY")
            if not self.endpoint:
                missing.append("AZURE_OPENAI_ENDPOINT")
            if not self.deployment_name:
                missing.append("AZURE_OPENAI_DEPLOYMENT_NAME")
                
            error_msg = f"Missing required Azure OpenAI credentials: {', '.join(missing)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
    
    async def get_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a completion using the Azure OpenAI GPT-4o model.
        
        Args:
            messages: List of message dictionaries in the format {"role": role, "content": content}
            temperature: Sampling temperature (higher = more random)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalty for token frequency
            presence_penalty: Penalty for token presence
            stop: List of stop sequences
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Response from the API
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                **kwargs
            )
            return response
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise
    
    async def get_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Get a chat response as a string.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Response content as a string
        """
        response = await self.get_completion(messages, **kwargs)
        return response