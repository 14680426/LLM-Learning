# 导入OpenAI客户端库，用于与vLLM服务进行交互
from openai import OpenAI
import json

# 创建OpenAI客户端实例，连接到本地运行的vLLM服务
client = OpenAI(
    api_key="xxx",  # 使用占位符，实际使用时可忽略或设置为任意值
    base_url="http://127.0.0.1:8000/v1"  # 指向本地vLLM服务的API地址
)

# 定义提示词，要求将一段话转换为学术论文风格
prompt = '''请将下面这段话润色成学术论文风格：
transformer 是一种用于序列到序列的深度学习模型，相较于传统的RNN 和 lstm ，它引入了注意力机制，能够更好的关注到序列中的语义信息，同时解决了长距离依赖问题，并且能够并行处理'''

print("发送的提示词:")
print(prompt)

# 调用模型生成响应，不启用技能模式
response_without_skill = client.chat.completions.create(
    model="Qwen3-4B-Thinking-2507",  # 使用本地部署的Qwen3-4B-Thinking-2507模型
    messages=[
        {'role': 'user', 'content': prompt}
    ],
    max_tokens=512,  # 将max_tokens设置为512，确保不超过模型的最大上下文长度1024
    temperature=0.9,  # 温度系数，控制输出随机性
    stream=False  # 不使用流式输出
)

print("\n完整响应对象:")
print(json.dumps(response_without_skill.model_dump(), indent=2, ensure_ascii=False))

# 检查响应对象的结构
print("\n响应对象类型:", type(response_without_skill))
if hasattr(response_without_skill, 'choices') and response_without_skill.choices:
    print("choices数量:", len(response_without_skill.choices))
    first_choice = response_without_skill.choices[0]
    print("第一个choice的属性:", dir(first_choice))
    if hasattr(first_choice, 'message'):
        print("message属性:", first_choice.message)
        if hasattr(first_choice.message, 'content'):
            print("content属性:", first_choice.message.content)
            print(f"普通提问的结果：\n{first_choice.message.content}")
        else:
            print("message对象没有content属性")
    else:
        print("choice对象没有message属性")
else:
    print("响应对象没有有效的choices")

# 打印普通提问的结果
if hasattr(response_without_skill, 'choices') and response_without_skill.choices:
    if hasattr(response_without_skill.choices[0], 'message') and hasattr(response_without_skill.choices[0].message, 'content'):
        print(f"\n普通提问的结果：\n{response_without_skill.choices[0].message.content}")
    else:
        print("\n无法从响应中提取内容")
else:
    print("\n响应对象中没有有效内容")