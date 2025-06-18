from agno.embedder.voyageai import VoyageAIEmbedder


class Embeddings:
    def __init__(self, model_name="voyage-code-3", directory="data/chunks"):
        """
        Initialize the Embeddings class with the specified model name.
        
        Args:
            model_name (str): The name of the embedding model to use.
        """
        self.embedder = VoyageAIEmbedder(id=model_name)

    def embed(self, chunks):
        """
        Embed a list of code chunks (each chunk is a json file) using the specified embedding model.
        Args:
            chunks (list): A list of code chunks to embed.
        Returns:
            list: A list of embeddings for the provided code chunks.
        """
        embeddings = []
        for chunk in chunks:
            embedding = self.embedder.embed(chunk)
            embeddings.append(embedding)
        return embeddings