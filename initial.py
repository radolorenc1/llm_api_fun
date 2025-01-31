from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
import sys

if len(sys.argv) < 2:
    print("Please provide a question as a command line argument.")
    print("Example: python initial.py 'What is the capital of France?'")
    sys.exit(1)

user_question = " ".join(sys.argv[1:])

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ.get('DEEPSEEK_API_KEY'),
)

completion = client.chat.completions.create(
  model="deepseek/deepseek-r1-distill-qwen-1.5b",
  messages=[
    {
      "role": "user",
      "content": user_question
    }
  ]
)
print(completion.choices[0].message.content)