# 加载第三方库
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

# 将JSON文件转换为Hugging Face Dataset格式
df = pd.read_json('../Datasets/huanhuan.json')  # 注意路径
ds = Dataset.from_pandas(df)

# 打印数据集基本信息
print(f"数据集大小: {len(ds)}")
print(f"数据集列名: {ds.column_names}")

# 加载模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained('/data/LLM Learning/Models/Qwen/Qwen3-4B-Thinking-2507', trust_remote=True)

# 构造一个示例chat template
messages = [
    {"role": "system", "content": "===system_message_test==="},
    {"role": "user", "content": "===user_message_test==="},
    {"role": "assistant", "content": "===assistant_message_test==="}
]

# 应用聊天模板，生成模型输入格式
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

# 输出格式化后的文本
print(text)