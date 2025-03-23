import openai

openai.api_key = 'your-api-key'

context = "Provide a detailed analysis of automated question generation techniques using generative models."

response = openai.Completion.create(
  engine="text-davinci-003", 
  prompt=f"Generate a question based on the following context: {context}",
  max_tokens=64,
  temperature=0.7
)

question = response['choices'][0]['text'].strip()
print(f"Generated Question: {question}")
