import tiktoken


class Tokenizer:
    def __init__(self, model_name = "cl100k_base", directory = "data/chunks"):
        """
        Initialize the tokenizer with the specified model name.
        
        Args:
            model_name (str): The name of the model to use for tokenization.
        """
        self.model_name = model_name
        self.directory = directory
        self.tokenizer = tiktoken.get_encoding(model_name)
        
    def tokenize_json(self, json_chunk):
        """
        Tokenize a JSON chunk and return the number of tokens.
        
        Args:
            json_chunk (dict): The JSON chunk to tokenize.
        
        Returns:
            int: The number of tokens in the JSON chunk.
        """
        json_str = str(json_chunk)
        tokens = self.tokenizer.encode(json_str)
        return len(tokens)
    
    def tokenize(self):
        """
        Tokenize all JSON chunks in the specified directory and return a list of token counts.
        
        Returns:
            list: A list of token counts for each JSON chunk.
        """
        import os
        import json
        
        token_counts = []
        
        for filename in os.listdir(self.directory):
            if filename.endswith(".json"):
                file_path = os.path.join(self.directory, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    json_chunk = json.load(f)
                    token_count = self.tokenize_json(json_chunk)
                    token_counts.append(token_count)
        
        return token_counts
    
    def token_stats(self):
        """
        Calculate and print statistics about the token counts.
        
        Returns:
            dict: A dictionary containing the total, average, minimum, and maximum token counts.
        """
        token_counts = self.tokenize()
        
        if not token_counts:
            return {
                "total_tokens": 0,
                "average_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0
            }
        
        total_tokens = sum(token_counts)
        average_tokens = total_tokens / len(token_counts)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        
        return {
            "total_tokens": total_tokens,
            "average_tokens": average_tokens,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens
        }
    
    