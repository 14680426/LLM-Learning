import torch
from peft import PeftModel  # 用于加载LoRA微调模型
from transformers import AutoTokenizer, AutoModelForCausalLM  # 用于加载基础模型和分词器

# 定义基础模型路径和LoRA微调模型路径
# 基础模型路径：指向原始预训练模型的存储位置
model_path = '/data/LLM Learning/Models/Qwen/Qwen3-4B-Thinking-2507'  # 基础模型参数路径
# LoRA微调模型路径：指向训练好的LoRA模型检查点位置
lora_path = '/data/LLM Learning/Finetune/output/Qwen3_4B-lora/checkpoint-100'  # 这里改成你的 LoRA 输出对应的 checkpoint 地址

# 加载分词器（Tokenizer）
# 使用AutoTokenizer从预训练模型中加载分词器，用于将文本转换为模型可处理的token序列
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载基础模型
# 使用AutoModelForCausalLM加载基础模型，设置设备映射为自动分配、使用bfloat16精度以提高效率
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True  # 允许加载远程代码（如自定义模型）
)

# 加载LoRA权重
# 使用PeftModel.from_pretrained将LoRA微调权重应用到基础模型上
# 这样可以实现对原始模型的轻量级微调，而不需要修改原始模型的所有参数
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 构造测试提示（Prompt）
# 设置一个简单的测试问题，用于验证模型是否正常工作
prompt = "你是谁？"

# 应用聊天模板（Chat Template）
# 将用户输入构造成符合模型要求的对话格式，包括系统消息、用户消息等
# add_generation_prompt=True 表示在输入末尾添加生成提示，以便模型知道需要生成回复
# tokenize=True 表示将文本转换为token序列
# return_tensors="pt" 表示返回PyTorch张量格式
# return_dict=True 表示返回字典格式的结果
# enable_thinking=False 表示不启用深度思考模式（如果模型支持的话）
inputs = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "假设你是皇帝身边的女人--甄嬛。"},
        {"role": "user", "content": prompt}
    ],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
    enable_thinking=False
)

# 将输入数据移动到与模型相同的设备上
# 修复设备不匹配问题：确保输入数据与模型在同一个设备上
for key in inputs:
    inputs[key] = inputs[key].to(model.device)

# 设置生成参数
# 配置模型生成文本时的行为参数
gen_kwargs = {
    "max_length": 2500,  # 最大生成长度（token数量）
    "do_sample": True,   # 是否使用采样方式生成文本（而非贪心解码）
    "top_k": 1           # 采样时只考虑概率最高的1个token
}

# 在无梯度模式下进行推理
# 使用torch.no_grad()上下文管理器禁用梯度计算，因为这是推理阶段而不是训练阶段
with torch.no_grad():
    # 调用模型的generate方法生成文本
    # **inputs表示将inputs字典中的所有键值对作为参数传递给generate方法
    # **gen_kwargs表示将gen_kwargs字典中的所有键值对作为参数传递给generate方法
    outputs = model.generate(**inputs, **gen_kwargs)
    
    # 提取生成结果中的输出序列
    # outputs[0]是生成的token序列，inputs["input_ids"]是输入序列
    # shape[1]表示输入序列的长度，用于确定输出序列中哪些部分是新生成的内容
    output = outputs[0, inputs["input_ids"].shape[1]:]
    
    # 解码生成的token序列，将其转换为可读的文本
    # skip_special_tokens=True表示跳过特殊token（如<im_start>、<im_end>等）以获得干净的输出
    print(tokenizer.decode(output, skip_special_tokens=True))
