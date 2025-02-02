from typing import List, Dict, Optional
from pathlib import Path
import requests
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from openai import OpenAI
from dotenv import load_dotenv 

load_dotenv(dotenv_path="local.env")                   
api_key = os.getenv('OPENROUTER_API_KEY')
print(f'api_key: {api_key} ')

class LongevityAdvisor:
    
    # def __init__(self):


    def get_response(self, query: str) -> str:
        """Generates response to user query."""
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        completion = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
                {
                "role": "user",
                "content": query
                }
            ]
        )
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content

# Example usage
if __name__ == "__main__":


    print("Setting up the AI Longevity Advisor...")
    advisor = LongevityAdvisor()

    print("\nWelcome to the AI Longevity Advisor. Ask me anything about related to longevity.")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\nAdvisor: Take care and remember - longevity is a marathon, not a sprint.")
            break

        response = advisor.get_response(user_input)
        print("\nAdvisor:", response)
