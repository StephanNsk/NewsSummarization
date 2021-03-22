from tqdm import tqdm
from rouge import Rouge
from typing import Dict, Collection
from transformers import MBartForConditionalGeneration, MBart50Tokenizer


def _rouge_score(predictions: Collection[str], targets: Collection[str]) -> Dict[str, float]:
    """
    Computing of rouge-1-f, rouge-2-f, rouge-l-f and its mean
    :param predictions: (Collection[str]) model's prediction summaries
    :param targets: (Collection[str]) true summaries
    :return: (Dict[str, float]) computed metrics
    """
    assert len(predictions) == len(targets), 'Length of the predictions and targets must be the same!'
    rouge = Rouge()
    scores = rouge.get_scores(predictions, targets)
    rouge_1_f = sum([score['rouge-1']['f'] for score in scores]) / len(scores)
    rouge_2_f = sum([score['rouge-2']['f'] for score in scores]) / len(scores)
    rouge_l_f = sum([score['rouge-l']['f'] for score in scores]) / len(scores)
    result = {'rouge_1_f': rouge_1_f, 'rouge_2_f': rouge_2_f, 'rouge_l_f': rouge_l_f}
    result['score'] = sum(result.values()) / len(result)
    return result


class Summarizer:

    def __init__(self, model_path: str, device: str = 'cuda') -> None:
        self.device = device
        self.model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)
        self.tokenizer = MBart50Tokenizer.from_pretrained(model_path)

    def generate(self, text: str) -> str:
        """
        Generation of summary for input text
        :param text: (str) text for summarization
        :return: (str) summary
        """
        preprocessed_text = text.lower().replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        tokenized_text = self.tokenizer(preprocessed_text, truncation=True, padding=True, return_tensors="pt")
        input_ids = tokenized_text['input_ids'].to(self.device)
        attention_mask = tokenized_text['attention_mask'].to(self.device)
        model_output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)[0]
        summary = self.tokenizer.decode(model_output, skip_special_tokens=True)
        return summary

    def evaluate(self, texts: Collection[str], titles: Collection[str]) -> Dict[str, float]:
        """
        Model evaluation. Compute rouge-1-f, rouge-2-f, rouge-l-f and its mean
        :param texts: (Collection[str]) input texts
        :param titles: (Collection[str]) labels for input texts
        :return: (Dict[str, float]) computed metrics
        """
        predictions = [self.generate(text) for text in tqdm(texts)]
        return _rouge_score(predictions, titles)
