# 加载第三方库
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

# 将JSON文件转换为Hugging Face Dataset格式
# 读取huanhuan.json文件，该文件包含约18000条对话数据
df = pd.read_json('../Datasets/huanhuan.json')  # 注意路径
ds = Dataset.from_pandas(df)

# 加载模型的tokenizer
# 使用Qwen3-4B-Thinking模型的tokenizer，该模型支持深度推理模式
tokenizer = AutoTokenizer.from_pretrained('/data/LLM Learning/Models/Qwen/Qwen3-4B-Thinking-2507', trust_remote_code=True)

# 数据预处理函数：将输入数据转换为模型可接受的格式
def process_func(example):
    # 设置最大序列长度为1024个token，这是模型的最大上下文长度
    MAX_LENGTH = 1024
    
    # 初始化输入、注意力掩码和标签列表
    input_ids, attention_mask, labels = [], [], []
    
    # 构造chat template
    # 使用特定格式包装系统消息、用户消息和助手消息
    # <im_start>和<im_end>是特殊标记，用于分隔不同类型的消息
    # 在助手消息中添加了"
    instruction = f"<im_start>system\n现在你要扮演皇帝身边的女人--甄嬛<im_end>\n" \
                  f"<im_start>user\n{example['instruction']}{example['input']}<im_end>\n" \
                  f"<im_start>assistant\n<think>\n</think>\n<im_end>\n"
    
    # 将instruction部分进行分词
    instruction_tokenized = tokenizer(instruction, add_special_tokens=False)
    input_ids = instruction_tokenized["input_ids"]
    
    # 将output部分进行分词
    response = tokenizer(example["output"], add_special_tokens=False)
    
    # 将instruction部分和response部分的input_ids拼接，末尾添加eos token作为标记结束的token
    input_ids = input_ids + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 生成attention_mask，表示哪些token是有效的
    attention_mask = [1] * len(input_ids)
    
    # 生成labels，用于训练时计算损失
    # 对于instruction部分，设置为-100表示这些位置不计算Loss（即模型不需要预测这部分）
    labels = [-100] * len(instruction_tokenized["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 如果长度超过最大长度，则截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 使用定义的函数对数据集进行处理
# 对数据集中的每个样本应用process_func函数进行处理，处理后移除原始数据列
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
print(f"数据集处理完成，共 {len(tokenized_id)} 个样本")

# 打印处理后的数据集信息
print("Dataset:")
print(tokenized_id)  # 显示数据集的基本信息，如样本数量、列信息等

# 显示处理后的数据示例
print("\n# 最终模型的输入")
# 解码第一个样本的input_ids，将其转换为可读的文本格式，展示模型实际接收到的输入
print(tokenizer.decode(tokenized_id[0]["input_ids"]))

# 显示最终模型的输出
print("\n# 最终模型的输出")
# 解码第一个样本的labels，过滤掉值为-100的位置（这些位置不计算损失），展示模型需要预测的输出
print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[0]["labels"]))))


