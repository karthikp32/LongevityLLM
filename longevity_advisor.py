from typing import List, Dict, Optional
import json
from pathlib import Path
import requests
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import traceback


class ContentCollector:
    """Collects and processes content from various sources."""
    
    def __init__(self, output_dir: str = "content"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def get_youtube_playlist_transcripts(self, playlist_url: str) -> List[str]:
        """Downloads and processes transcripts from a YouTube playlist."""
        try:
            response = requests.get(playlist_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            video_ids = [
                link['href'].split('v=')[1].split('&')[0]
                for link in soup.find_all('a', href=True)
                if 'watch?v=' in link['href']
            ]
            print(f'video_ids: {video_ids}')
            transcripts = []
            for video_id in video_ids:
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    full_text = " ".join([entry['text'] for entry in transcript])
                    transcripts.append(full_text)
                except Exception as e:
                    print(f"Error getting transcript for video ID {video_id}: {e}")
            return transcripts
        except Exception as e:
               # Capture the full traceback of the error
            error_message = traceback.format_exc()
            print(f"Error processing playlist {playlist_url}: {e}")
                # Optionally, print the full traceback (this shows where the exception occurred)
            print(f"Detailed traceback:\n{error_message}")
            return []

    def scrape_articles(self, website_url: str) -> List[str]:
        """Scrapes and processes all articles from a website."""
        try:
            response = requests.get(website_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            article_links = [
                link['href'] for link in soup.find_all('a', href=True)
                if '/article' in link['href']
            ]
            articles = []
            for article_url in article_links:
                try:
                    full_url = article_url if article_url.startswith('http') else website_url + article_url
                    article_response = requests.get(full_url)
                    article_soup = BeautifulSoup(article_response.text, 'html.parser')
                    content = article_soup.find('article') or article_soup.find('main')
                    if content:
                        articles.append(content.get_text(strip=True))
                except Exception as e:
                    print(f"Error scraping article {article_url}: {e}")
            return articles
        except Exception as e:
            print(f"Error scraping website {website_url}: {e}")
            return []

    def process_content(self, content: str, source: str) -> None:
        """Processes and saves content with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{source}.txt"
        
        with open(self.output_dir / filename, 'w', encoding='utf-8') as f:
            f.write(content)

# Other classes remain unchanged...
class KnowledgeBaseBuilder:
    """Builds and manages the knowledge base using sentence transformers."""
    
    def __init__(self, content_dir: str):
        self.content_dir = Path(content_dir)
        # Use BERT-based model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = 1000
        self.overlap = 200
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    def process_documents(self) -> tuple:
        """Processes all documents into chunks and embeddings."""
        chunks = []
        for file in self.content_dir.glob("*.txt"):
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks.extend(self.chunk_text(text))
        
        # Create embeddings
        embeddings = self.model.encode(chunks)
        return chunks, embeddings

class LocalLLMInterface:
    """Interfaces with Ollama for local LLM inference."""
    
    def __init__(self, model_name: str = "mistral"):
        """Initialize with specified model. Default is Mistral."""
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"
        
        # Ensure model is installed
        self._ensure_model()
    
    def _ensure_model(self):
        """Checks if model is available locally, downloads if not."""
        try:
            response = requests.post(
                f"{self.base_url}/pull",
                json={"name": self.model_name}
            )
            if response.status_code == 200:
                print(f"Model {self.model_name} is ready")
            else:
                print(f"Error preparing model: {response.text}")
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("Please ensure Ollama is installed and running")
            
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generates response using the local LLM."""
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"Context: {context}\n\nQuestion: {prompt}\n\nProvide a detailed response based on the context and your knowledge of longevity medicine:",
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error generating response: {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

class LongevityAdvisor:
    """Main advisor class using local LLM and knowledge base."""
    
    def __init__(self, chunks: List[str], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings
        self.llm = LocalLLMInterface()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.system_prompt = """
        You are an expert longevity advisor with deep knowledge of Peter Attia's approach to health optimization. 
        Base your responses on the provided context and scientific evidence.
        Focus on:
        1. The "Four Horsemen" of longevity: glycemic control, inflammation, exercise, and sleep
        2. Specific, actionable recommendations
        3. Relevant biomarkers and testing
        4. Breaking down complex topics
        
        Provide educational information while being clear you're not giving medical advice.
        """
    
    def get_relevant_context(self, query: str, n_chunks: int = 3) -> str:
        """Retrieves most relevant context chunks for the query."""
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top chunks
        top_indices = similarities.argsort()[-n_chunks:][::-1]
        relevant_chunks = [self.chunks[i] for i in top_indices]
        
        return "\n\n".join(relevant_chunks)

    def get_response(self, query: str) -> str:
        """Generates response to user query."""
        try:
            # Get relevant context
            context = self.get_relevant_context(query)
            
            # Combine with system prompt
            full_prompt = f"{self.system_prompt}\n\n{query}"
            
            # Get response from local LLM
            response = self.llm.generate_response(full_prompt, context)
            
            # Add disclaimer if needed
            if self._needs_medical_disclaimer(query, response):
                response += "\n\nNote: This information is educational only and not medical advice. Please consult with your healthcare provider for personalized recommendations."
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."

    def _needs_medical_disclaimer(self, query: str, response: str) -> bool:
        """Determines if response needs medical disclaimer."""
        medical_terms = ['disease', 'condition', 'treatment', 'medication', 'diagnosis', 'symptoms']
        return any(term in query.lower() or term in response.lower() for term in medical_terms)
# Updated `setup_advisor` function
def setup_advisor(playlist_url: str, website_url: str) -> LongevityAdvisor:
    """Sets up the complete advisor system."""
    # Initialize content collector
    collector = ContentCollector()

    # Collect content from YouTube playlist
    print("Collecting YouTube playlist transcripts...")
    transcripts = collector.get_youtube_playlist_transcripts(playlist_url)
    for idx, transcript in enumerate(transcripts):
        collector.process_content(transcript, f"youtube_video_{idx}")

    # Collect content from the website
    print("Scraping articles from the website...")
    articles = collector.scrape_articles(website_url)
    for idx, article in enumerate(articles):
        collector.process_content(article, f"website_article_{idx}")

    # Build knowledge base
    kb_builder = KnowledgeBaseBuilder("content")
    chunks, embeddings = kb_builder.process_documents()

    # Initialize advisor
    advisor = LongevityAdvisor(chunks, embeddings)

    return advisor

# Example usage
if __name__ == "__main__":
    # Provide source URLs
    playlist_url = "https://www.youtube.com/playlist?list=PLlFlZLYiJ88Pnq_MSHfRH5KsX07XXTdL_"
    website_url = "https://peterattiamd.com/"

    print("Setting up the AI Longevity Advisor...")
    advisor = setup_advisor(playlist_url, website_url)

    print("\nWelcome to the AI Longevity Advisor. Ask me anything about health optimization and longevity.")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\nAdvisor: Take care and remember - longevity is a marathon, not a sprint.")
            break

        response = advisor.get_response(user_input)
        print("\nAdvisor:", response)
