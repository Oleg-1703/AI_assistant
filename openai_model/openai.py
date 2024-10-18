import openai
api_key = 'api_key'
class OpenAIGPTAPI:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_response(self, prompt, model="gpt-4", max_tokens=150):
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()
