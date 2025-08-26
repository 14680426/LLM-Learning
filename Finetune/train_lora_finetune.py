# LoRA微调脚本
# 使用transformers框架进行监督式微调(SFT) with LoRA

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
import pandas as pd

# 首先配置 LoRA 参数
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 任务类型为 CLM，即 SFT 任务的类型
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 目标模块，即需要进行 LoRA 微调的模块
    inference_mode=False,  # 训练模式
    r=8,  # LoRA 秩，即 LoRA 微调的维度
    lora_alpha=32,  # LoRA alph，具体作用参考 LoRA 原理
    lora_dropout=0.1  # Dropout 比例
)

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    '/data/LLM Learning/Models/Qwen/Qwen3-4B-Thinking-2507', 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

# 打印原始模型参数信息
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"原始模型参数: {total_params / 1e9:.2f}B (十亿) 参数")
print(f"原始模型可训练参数: {trainable_params / 1e9:.2f}B (十亿) 参数")

# 启用梯度检查点降低显存占用
model.enable_input_require_grads()

# 获取 LoRA 微调后的模型
model = get_peft_model(model, config)

print("Config: ", config)

# 查看 LoRA 微调的模型参数
model.print_trainable_parameters()

# 获取LoRA模型参数信息
lora_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LoRA模型参数: {total_params / 1e9:.2f}B (十亿) 参数")
print(f"LoRA模型可训练参数: {lora_trainable_params / 1e6:.2f}M (百万) 参数")

# 打印模型数据类型
print(f"模型数据类型: {model.dtype}")

# 加载数据集和分词器
df = pd.read_json('../Datasets/huanhuan.json')  # 注意路径
ds = Dataset.from_pandas(df)

# 加载模型的tokenizer
tokenizer = AutoTokenizer.from_pretrained('/data/LLM Learning/Models/Qwen/Qwen3-4B-Thinking-2507', trust_remote_code=True)

# 数据预处理函数
def process_func(example):
    # 设置最大序列长度为1024(tokenizer最大限制)
    MAX_LENGTH = 1024
    
    # 构造对话模板
    instruction = f"<im_start>system\n现在你要扮演皇帝身边的女人--甄嬛<im_end>\n" \
                  f"<im_start>user\n{example['instruction']}{example['input']}<im_end>\n" \
                  f"<im_start>assistant\n{example['output']}<im_end>\n"
    tokenized = tokenizer(instruction, return_tensors='pt', padding=False, truncation=False)
    input_ids = tokenized['input_ids'][0].tolist()  # 转换为列表
    attention_mask = tokenized['attention_mask'][0].tolist()  # 转换为列表
    labels = tokenized['input_ids'][0].tolist()  # 转换为列表
    
    # 验证三个序列长度一致
    assert len(input_ids) == len(attention_mask) == len(labels)
    
    # 处理长度限制
    if len(input_ids) > MAX_LENGTH:
        # 截断前面部分，保留最新内容
        input_ids = input_ids[-MAX_LENGTH:]
        attention_mask = attention_mask[-MAX_LENGTH:]
        labels = labels[-MAX_LENGTH:]
    else:
        # 长度不足时填充到MAX_LENGTH
        padding_length = MAX_LENGTH - len(input_ids)
        input_ids.extend([tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        labels.extend([-100] * padding_length)
    
    # 返回处理后的数据
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# 预处理数据集
tokenized_id = ds.map(process_func, batched=False)

# 划分训练集和验证集
tokenized_id = tokenized_id.train_test_split(test_size=0.1)
train_dataset = tokenized_id['train']
eval_dataset = tokenized_id['test']

# 配置训练参数
args = TrainingArguments(
    output_dir="./output/Qwen3_4B-lora",  # 输出目录
    per_device_train_batch_size=3,  # 每个设备的训练批量大小
    per_device_eval_batch_size=3,   # 每个设备的评估批量大小
    gradient_accumulation_steps=2,  # 梯度累积步数
    logging_steps=10,  # 每10步打印一次日志
    num_train_epochs=3,  # 训练轮数
    save_steps=100,  # 每100步保存一次模型
    save_total_limit=3,  # 最多保存3个模型
    eval_strategy="steps",  # 每隔一定步数进行评估
    eval_steps=50,  # 每50步进行一次评估
    learning_rate=1e-4,  # 学习率
    warmup_steps=500,  # 预热步数
    weight_decay=0.01,  # 权重衰减
    fp16=False,  # 禁用FP16以避免梯度缩放问题
    bf16=True,  # 启用BF16训练
    max_grad_norm=0.0,  # 禁用梯度裁剪以避免BFloat16兼容性问题
    dataloader_pin_memory=False,  # 禁用pin_memory以避免潜在的内存问题
    save_on_each_node=True,  # 是否在每个节点上保存模型
    gradient_checkpointing=True,  # 是否使用梯度检查点
    report_to="none",  # 不使用任何报告工具
    push_to_hub=False,  # 不推送至Hugging Face Hub
    load_best_model_at_end=True,  # 训练结束后加载最佳模型
    metric_for_best_model="eval_loss",  # 根据评估损失选择最佳模型
    greater_is_better=False  # 评估指标越小越好
)

# 创建Trainer并配置SFT训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,  # 使用划分后的训练集
    eval_dataset=eval_dataset,    # 使用划分后的验证集
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[]
)

# 开始训练
trainer.train()