from git import Repo
import os


class GitFetcher:
    def __init__(self, repo_url, save_dir=None):
        self.repo_url = repo_url
        self.save_dir = save_dir or os.path.join(os.getcwd(), "./repos")
        
    def fetch_repo(self):
        """
        Fetch the repository from the given URL and save it to the specified directory.
        
        Returns:
            str: The path to the cloned repository.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        repo_name = self.repo_url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(self.save_dir, repo_name)
        
        if os.path.exists(repo_path):
            print(f"Repository {repo_name} already exists at {repo_path}. Pulling latest changes.")
            repo = Repo(repo_path)
            repo.remotes.origin.pull()
        else:
            print(f"Cloning repository {self.repo_url} into {repo_path}.")
            repo = Repo.clone_from(self.repo_url, repo_path)
        
        return repo_path