from typing import List, Dict, Optional
from pathlib import Path
import requests
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI


class LongevityAdvisor:
    
    # def __init__(self):


    def get_response(self, query: str) -> str:
        """Generates response to user query."""
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
        )
        
        completion = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {
            "role": "user",
            "content": "How many minutes of Zone 2 cardio should I do per week?"
            }
        ]
)
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content

# Example usage
if __name__ == "__main__":


    print("Setting up the AI Longevity Advisor...")
    advisor = LongevityAdvisor()

    print("\nWelcome to the AI Longevity Advisor. Ask me anything about Zone 2 training.")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\nAdvisor: Take care and remember - longevity is a marathon, not a sprint.")
            break

        response = advisor.get_response(user_input)
        print("\nAdvisor:", response)
