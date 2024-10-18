import torch
from transformers import BartTokenizer, BartForSequenceClassification, DistilBertTokenizer, DistilBertForQuestionAnswering

class BARTClassifier:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.model = BartForSequenceClassification.from_pretrained('facebook/bart-large-finetuned-squad')

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits.argmax(dim=1).item()  # Предположим, что классы 0 и 1

class DistilBERTQASummarizer:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

    def summarize(self, question, context):
        inputs = self.tokenizer.encode_plus(question, context, return_tensors='pt')
        with torch.no_grad():
            start_scores, end_scores = self.model(**inputs)
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores) + 1  # Плюс 1, чтобы включить последний токен
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_index:end_index]))

