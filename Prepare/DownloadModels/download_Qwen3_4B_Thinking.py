from modelscope import snapshot_download, AutoModel, AutoTokenizer

model_dir = snapshot_download('Qwen/Qwen3-4B-Thinking-2507', cache_dir='/data/LLM Learning/Models', revision='master')