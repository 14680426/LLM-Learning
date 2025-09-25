import logging
import os
# 导入必要的transformers库组件，用于加载和使用预训练语言模型
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置模型本地路径，指定预训练模型的存储位置
model_name = "/data/LLM Learning/Models/Qwen/Qwen3-4B-Thinking-2507"

# 加载分词器和模型
# 分词器用于将文本转换为模型可以理解的token序列
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 模型用于生成文本，是因果语言模型(Causal Language Model)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",  # 自动选择合适的数据类型以优化性能
    device_map="cuda:0",  # 指定使用GPU设备进行计算，提高推理速度
    trust_remote_code=True  # 信任并允许执行模型中的自定义代码
)

def chat_with_local_model(model, tokenizer, user_input: str, history: list = None, temperature: float = 0.7, max_new_tokens: int = 1024, enable_thinking: bool = True) -> dict:
    """
    使用本地大模型进行对话的函数

    Args:
        model: 加载好的模型
        tokenizer: 加载好的分词器
        history: 历史对话记录列表，格式为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        temperature: 温度参数，控制输出的随机性（0.0~1.0）
        max_new_tokens: 最大生成token数量
        enable_thinking: 是否启用深度推理模式

    Returns:
        dict: 包含思考内容和最终回答的字典 {"thinking_content": str, "content": str}
    """

    # 初始化历史记录
    if history is None:
        history = []

    # 构建消息列表
    messages = []

    # 添加历史对话
    for msg in history:
        messages.append(msg)

    # 添加当前用户输入
    messages.append({"role": "user", "content": user_input})

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking  # 选择是否打开深度推理模式
    )

    # 将输入文本转换为模型可处理的张量格式
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成文本
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,  # 设置最大生成token数量
        temperature=temperature,
        do_sample=True if temperature > 0 else False  # 当temperature > 0时启用采样
    )

    # 提取新生成的token ID
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # 解析思考内容
    try:
        # 查找结束标记
        thinking_end = full_response.find("</think>")
        
        # 如果找到思考标记
        if "<|thinking|>" in full_response and thinking_end != -1:
            thinking_start = full_response.index("<|thinking|>") + len("<|thinking|>")
            
            # 提取思考内容
            thinking_content = full_response[thinking_start:thinking_end].strip()
            
            # 提取最终回答
            content = full_response[thinking_end + len("</think>"):].strip()
        else:
            # 如果没有找到思考标记，将全部内容作为回答
            thinking_content = ""
            content = full_response.strip()
            
        # 处理内容中的特殊标记
        if "<|startofthink|>" in content:
            content_start = full_response.index("<|startofthink|>") + len("<|startofthink|>")
            content = full_response[content_start:].strip()
            
    except Exception as e:
        logger.warning(f"Failed to parse thinking content: {str(e)}")
        thinking_content = ""
        content = full_response.strip()
    else:
        # 不启用思考模式时的处理
        thinking_content = ""
        content = full_response.strip()
        
        # 移除可能存在的特殊标记
        for tag in ["<|startofthink|>", "<|thinking|>", "</think>"]:
            if tag in content:
                content = content.split(tag)[-1].strip()

    return {
        "thinking_content": thinking_content,
        "content": content
    }

# 使用示例
if __name__ == "__main__":
    # 单轮对话示例
    try:
        result = chat_with_local_model(
            model=model,
            tokenizer=tokenizer,
            user_input="你好，请介绍一下自己",
            temperature=0.7
        )

        print("\n================== 单轮对话结果 ==================")
        print("thinking content:", result["thinking_content"])
        print("content:", result["content"])

        # 多轮对话示例
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！我是Qwen，一个AI助手。"}
        ]

        result = chat_with_local_model(
            model=model,
            tokenizer=tokenizer,
            user_input="你能帮我写一首诗吗？",
            history=history,
            temperature=0.8
        )

        print("\n================== 多轮对话结果 ==================")
        print("thinking content:", result["thinking_content"])
        print("content:", result["content"])
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
