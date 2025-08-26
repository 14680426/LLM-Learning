# 导入requests库，用于发送HTTP请求
import requests

# 设置API端点URL，指向本地运行的vLLM服务
url = "http://127.0.0.1:8000/v1/chat/completions"

# 定义请求载荷(payload)
payload = {
    # 指定要使用的模型名称
    "model": "Qwen3-4B-Thinking-2507",
    # 定义对话消息列表
    "messages": [
        {
            # 指定角色为用户
            "role": "user",
            # 用户的具体问题内容
            "content": "请详细介绍一下你自己～"
        }
    ]
}

# 设置请求头部信息
headers = {
    # 认证信息，需要将<token>替换为实际的API令牌
    "Authorization": "Bearer <token>",
    # 指定内容类型为JSON格式
    "Content-Type": "application/json"
}

# 发送POST请求到vLLM服务，并获取响应
response = requests.post(url, json=payload, headers=headers)

# 打印响应的JSON内容
print(response.json())