from openai import OpenAI

client = OpenAI(api_key="xxx",
                base_url="http://127.0.0.1:8000/v1")
response = client.chat.completions.create(
    model="Qwen3-4B-Thinking-2507",
    messages=[
        {'role': 'user', 'content': '你好哇'}
    ],
    max_tokens=512,
    temperature=0.7,
    stream=False
)
print(response.choices[0].message)