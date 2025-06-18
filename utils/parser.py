import os
import ast
import re
from pathlib import Path
import json


class Parser:
    def __init__(self, repo_path, chunk_dir=None):
        self.repo_path = Path(repo_path)
        self.supported_extensions = {'.py', '.js'}
        self.chunk_dir = chunk_dir or Path("data/chunks")
        self.chunk_counter = 0
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        if not self.repo_path.is_dir():
            raise ValueError(f"The provided path {repo_path} is not a valid directory.")

    def get_all_source_files(self):
        """Recursively get all .py and .js files."""

        repo = Path(self.repo_path)
        return [
            file for file in repo.rglob('*') if file.suffix in self.supported_extensions
        ]
        
    def save_chunk(self, chunk):
        """Save a single chunk to the data/chunks/ folder."""
        self.chunk_counter += 1
        chunk_path = self.chunk_dir / f"chunk_{self.chunk_counter:04}.json"
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2)

        
    def parse_python_file(self, file_path):
        """Parse a Python file and return its AST."""
        with open(file_path, "r", encoding="utf-8") as file:
            source = file.read()
            
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return None
        
        chunks = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                start_line = node.lineno
                end_line = getattr(node, "end_lineno", None)

                code_lines = source.splitlines()
                code_chunk = "\n".join(code_lines[start_line - 1:end_line]) if end_line else ""

                chunks.append({
                    "language": "python",
                    "type": node.__class__.__name__.lower(),
                    "name": node.name,
                    "filepath": str(file_path),
                    "start_line": start_line,
                    "code": code_chunk
                })
                
        # Save each chunk to a file
        for chunk in chunks:
            self.save_chunk(chunk)
                
        
    
    def parse_javascript_file(self, file_path):
        """Parse a JavaScript file and return its functions and classes."""
        with open(file_path, "r", encoding="utf-8") as file:
            source = file.read()
        
        # Simple regex to find function and class definitions
        pattern = r"(function\s+\w+\s*\([^)]*\)\s*{)|(\w+\s*=\s*\([^)]*\)\s*=>\s*{)"
        matches = list(re.finditer(pattern, source))
        
        chunks = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
            code_chunk = source[start:end]

            chunks.append({
                "language": "javascript",
                "type": "function",
                "name": match.group().split()[1] if "function" in match.group() else "anonymous_arrow",
                "filepath": str(file_path),
                "start_char": start,
                "code": code_chunk
            })

        # Save each chunk to a file
        for chunk in chunks:
            self.save_chunk(chunk)
    
    def parse_repo(self):
        """Parse all source files in the repository and return their chunks."""
        code_chunks = []
        files = self.get_all_source_files()
        
        for file in files:
            ext = file.suffix
            if ext == '.py':
                self.parse_python_file(file)
            elif ext == '.js':
                self.parse_javascript_file(file)
            else:
                continue
            
        print(f"Parsed {len(code_chunks)} code chunks from the repository.")
            