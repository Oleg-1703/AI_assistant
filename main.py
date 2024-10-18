from bart_model.bart_classifier import BARTModel
from openai_model.openai import OpenAIGPTAPI
api_key = 'api_key'
# Инициализация моделей
bart_model = BARTModel("facebook/bart-large-mnli", "facebook/bart-large-cnn")
openai_api = OpenAIGPTAPI(api_key)

def handle_student_query(question, age, academic_performance, neurodiversity):
    # Классификация вопроса
    class_id = bart_model.classify_question(question)

    # Суммаризация вопроса
    summarized_question = bart_model.summarize_text(question)

    # Подготовка промпта
    prompt = bart_model.prepare_prompt(summarized_question, age, academic_performance, neurodiversity)

    # Получение ответа от OpenAI
    response = openai_api.generate_response(prompt)
    
    return response

# Пример использования
if __name__ == "__main__":
    age = 12
    academic_performance = {"Math": "B", "English": "A"}
    neurodiversity = "ADHD"
    question = "Как мне лучше подготовиться к экзамену по математике?"

    answer = handle_student_query(question, age, academic_performance, neurodiversity)
    print(answer)