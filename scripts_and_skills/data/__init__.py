# scripts_and_skills/data — local data layer
# Parquet-based storage for prompts, conversations, embeddings
# Import directly from submodules to avoid eager loading (heavy deps):
#   from scripts_and_skills.data.prompt_store    import PromptStore
#   from scripts_and_skills.data.embeddings      import EmbeddingStore
#   from scripts_and_skills.data.dataset_generator import DatasetGenerator
#   from scripts_and_skills.data.arxiv_crawler   import ArxivCrawler
#   from scripts_and_skills.data.classifier      import Classifier
#   from scripts_and_skills.data.web_search      import search, fetch_url_text
