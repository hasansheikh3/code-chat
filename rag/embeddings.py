from agno.embedder.voyageai import VoyageAIEmbedder
import os
import json
import dotenv
from typing import List, Dict, Any

class Embeddings:
    def __init__(self, model_name="voyage-code-3", directory="data/chunks"):
        """
        Initialize the Embeddings class with the specified model name.
        Args:
            model_name (str): The name of the embedding model to use.
            directory (str): Path to the directory containing chunk JSON files.
        """
        dotenv.load_dotenv()
        api_key = os.getenv("VOYAGEAI_API_KEY")
        if not api_key:
            raise ValueError("VOYAGEAI_API_KEY not found in environment variables")
        
        self.embedder = VoyageAIEmbedder(id=model_name, api_key=api_key)
        self.directory = directory
        self.chunks = []
        self.embeddings = []
        
    def load_chunks(self) -> List[Dict[str, Any]]:
        """
        Load all JSON chunk files from the specified directory.
        Returns:
            list: A list of loaded chunk dictionaries.
        """
        chunks = []
        
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"Directory {self.directory} does not exist")
        
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(self.directory) if f.endswith('.json')]
        json_files.sort()  # Sort to ensure consistent ordering
        
        print(f"Found {len(json_files)} chunk files")
        
        for filename in json_files:
            filepath = os.path.join(self.directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    chunk_data['chunk_file'] = filename  # Add filename for reference
                    chunks.append(chunk_data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        self.chunks = chunks
        print(f"Successfully loaded {len(chunks)} chunks")
        return chunks
    
    def prepare_text_for_embedding(self, chunk: Dict[str, Any]) -> str:
        """
        Prepare chunk data for embedding by creating a comprehensive text representation.
        Args:
            chunk (dict): A single chunk dictionary.
        Returns:
            str: Formatted text ready for embedding.
        """
        # Create a comprehensive text representation
        parts = []
        
        # Add metadata context
        if chunk.get('type'):
            parts.append(f"Type: {chunk['type']}")
        if chunk.get('name'):
            parts.append(f"Name: {chunk['name']}")
        if chunk.get('filepath'):
            parts.append(f"File: {chunk['filepath']}")
        if chunk.get('language'):
            parts.append(f"Language: {chunk['language']}")
        
        # Add the actual code
        if chunk.get('code'):
            parts.append("Code:")
            parts.append(chunk['code'])
        
        return "\n".join(parts)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Embed a list of code chunks using the specified embedding model.
        Args:
            chunks (list, optional): A list of code chunks to embed. 
                                   If None, uses loaded chunks.
        Returns:
            list: A list of dictionaries containing chunk data and embeddings,
                  ready for vector store insertion.
        """
        if chunks is None:
            if not self.chunks:
                print("No chunks loaded. Loading chunks first...")
                self.load_chunks()
            chunks = self.chunks
        
        if not chunks:
            raise ValueError("No chunks available for embedding")
        
        print(f"Embedding {len(chunks)} chunks...")
        embeddings_data = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Prepare text for embedding
                text_to_embed = self.prepare_text_for_embedding(chunk)
                
                # Generate embedding vector
                embedding_vector = self.embedder.embed(text_to_embed)
                
                # Create vector store ready format
                embedding_item = {
                    'id': f"chunk_{i:04d}",  # Unique ID for vector store
                    'vector': embedding_vector,  # The actual embedding vector
                    'metadata': {
                        'type': chunk.get('type', ''),
                        'name': chunk.get('name', ''),
                        'filepath': chunk.get('filepath', ''),
                        'language': chunk.get('language', ''),
                        'start_char': chunk.get('start_char', 0),
                        'chunk_file': chunk.get('chunk_file', ''),
                        'text': text_to_embed  # Full text for reference
                    },
                    'code': chunk.get('code', '')  # Keep original code separate
                }
                
                embeddings_data.append(embedding_item)
                
                if (i + 1) % 5 == 0:  # Progress update every 5 chunks
                    print(f"Processed {i + 1}/{len(chunks)} chunks")
                    
            except Exception as e:
                print(f"Error embedding chunk {i}: {e}")
                # Skip failed embeddings rather than breaking the process
                continue
        
        self.embeddings = embeddings_data
        print(f"Successfully embedded {len(embeddings_data)} chunks")
        return embeddings_data
    
    def get_embeddings_for_vector_store(self) -> List[Dict[str, Any]]:
        """
        Get embeddings in a format ready for vector store insertion.
        Returns:
            list: List of dictionaries with 'id', 'vector', and 'metadata' keys.
        """
        if not self.embeddings:
            raise ValueError("No embeddings available. Run embed_chunks() first.")
        
        return self.embeddings
    
    def save_embeddings(self, output_path: str = "data/embeddings.json"):
        """
        Save embeddings to a JSON file.
        Args:
            output_path (str): Path where to save the embeddings.
        """
        if not self.embeddings:
            raise ValueError("No embeddings to save. Run embed_chunks() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert embeddings to serializable format
        serializable_embeddings = []
        for item in self.embeddings:
            serializable_item = {
                'chunk_data': item['chunk_data'],
                'embedding': item['embedding'].tolist() if item.get('embedding') is not None else None,
                'text_used': item.get('text_used', ''),
                'error': item.get('error')
            }
            serializable_embeddings.append(serializable_item)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_embeddings, f, indent=2, ensure_ascii=False)
        
        print(f"Embeddings saved to {output_path}")
    
    def load_embeddings(self, input_path: str = "data/embeddings.json"):
        """
        Load embeddings from a JSON file.
        Args:
            input_path (str): Path to load embeddings from.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Embeddings file {input_path} not found")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            loaded_embeddings = json.load(f)
        
        # Convert back to proper format (lists back to numpy arrays if needed)
        self.embeddings = loaded_embeddings
        print(f"Loaded {len(loaded_embeddings)} embeddings from {input_path}")
        return loaded_embeddings