import torch
from transformers import BartForSequenceClassification, BartTokenizer, BartForConditionalGeneration

class BARTModel:
    def __init__(self, model_name_classification, model_name_summarization):
        # Инициализация токенизатора и моделей
        self.tokenizer_classifier = BartTokenizer.from_pretrained(model_name_classification)
        self.model_classifier = BartForSequenceClassification.from_pretrained(model_name_classification)
        
        self.tokenizer_summarizer = BartTokenizer.from_pretrained(model_name_summarization)
        self.model_summarizer = BartForConditionalGeneration.from_pretrained(model_name_summarization)

    def classify_question(self, question):
        inputs = self.tokenizer_classifier(question, return_tensors="pt")
        outputs = self.model_classifier(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        return predicted_class

    def summarize_text(self, text):
        inputs = self.tokenizer_summarizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model_summarizer.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0)
        summarized_text = self.tokenizer_summarizer.decode(summary_ids[0], skip_special_tokens=True)
        return summarized_text

    def prepare_prompt(self, question, age=None, academic_performance=None, neurodiversity=None):
        prompt = f"Question: {question}\n"
        if age:
            prompt += f"Age: {age}\n"
        if academic_performance:
            prompt += "Academic Performance:\n"
            for subject, grade in academic_performance.items():
                prompt += f"- {subject}: {grade}\n"
        if neurodiversity:
            prompt += f"Neurodiversity: {neurodiversity}\n"
        return prompt
