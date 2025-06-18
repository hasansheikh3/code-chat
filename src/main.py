# the structure of the project for now (v1):
# User enters a URL to github repository
# The program will fetch the repository
# The program will parse the repository
# Maybe use an AST Parser to parse the code
# The program will create chunks from the parsed code
# Those chunks will be embedded using Voyage AI model
# The program will store the chunks in a vector database
# The user can then query the codebase and get answers

import os
from utils.llms import LLM
from utils.git import GitFetcher
from utils.parser import Parser
from rag.tokenizer import Tokenizer
from rag.embeddings import Embeddings

repo_url = input("Enter the GitHub repository URL: ")
if not repo_url:
    raise ValueError("Repository URL cannot be empty.")
# if not repo_url ends with .git, append .git to the end
if not repo_url.endswith(".git"):
    repo_url += ".git"
    
# Fetch the repository
git_fetcher = GitFetcher(repo_url)
repo_path = git_fetcher.fetch_repo()
print(f"Repository cloned to: {repo_path}")


# Parse the repository
parser = Parser(repo_path)
parser.parse_repo()
print("Repository parsed successfully.")


print("Tokenizing the codebase...")
tokenizer = Tokenizer()
token_counts = tokenizer.token_stats()
print(f"Token counts: {token_counts}")

# if max token count is greater than 8000, we need to chunk it (for now just log it)
if token_counts['max_tokens'] > 8000:
    print("Warning: Some chunks exceed the maximum token limit of 8000. Consider chunking them further.")
    
