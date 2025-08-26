from modelscope import snapshot_download, AutoModel, AutoTokenizer

model_dir = snapshot_download('Qwen/Qwen3-4B', cache_dir='/data/LLM Learning/Models', revision='master')