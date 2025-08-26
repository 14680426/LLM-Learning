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
    torch_dtype="auto",  # 自动选择合适的数据类型以优化性能
    device_map="cuda:0",  # 指定使用GPU设备进行计算，提高推理速度
    trust_remote_code=True  # 信任并允许执行模型中的自定义代码
)

# 准备模型输入，构造对话消息
prompt = "你好"  # 用户输入的提示词
messages = [
    # 构造消息列表，指定角色为"user"表示用户输入
    {"role": "user", "content": prompt}
]
# 应用聊天模板，将消息格式化为模型所需的输入格式
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,  # 不直接进行分词，返回格式化后的文本
    add_generation_prompt=True,  # 添加生成提示，告诉模型需要生成回复
    enable_thinking=True  # 启用深度推理模式，允许模型进行更复杂的思考
)
print("*"*10, "输入文本", "*"*10)
print(text)  # 打印格式化后的输入文本

# 将输入文本转换为模型可处理的张量格式
# return_tensors="pt"表示返回PyTorch张量
# to(model.device)将张量移动到模型所在的设备(GPU/CPU)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("*"*10, "输入张量", "*"*10)
print(model_inputs)

# 生成文本，使用模型基于输入内容生成回复
generated_ids = model.generate(
    **model_inputs,  # 解包模型输入参数
    max_new_tokens=32768  # 设置最大生成token数量，控制生成文本的长度
)

# 提取新生成的token ID，去掉输入部分，只保留模型生成的部分
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# 解析思考内容，分离模型的推理过程和最终回答
try:
    # 查找结束标记151668(<tool_call>)的位置，该标记分隔思考内容和最终回答
    # 通过反向查找索引，然后用总长度减去反向索引得到正向索引位置
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    # 如果没有找到结束标记，则将索引设为0
    index = 0

# 解码思考内容和最终回答
# skip_special_tokens=True表示跳过特殊token，只解码普通文本
# strip("\n")去除首尾换行符
thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")  # 思考内容
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")  # 最终回答

# 打印结果，分别输出模型的思考过程和最终回答内容
print("thinking content:", thinking_content)
print("content:", content)