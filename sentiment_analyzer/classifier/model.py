import json
import torch
# import torch.nn.functional as F
from transformers import AutoTokenizer

from .sentiment_classifier import SentimentClassifier

int2Label = {0: 'anxiety',
            1: 'depression',
            2: 'normal',
            3: 'ocd',
            4: 'other',
            5: 'suicide'
} 
with open("config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(config["BERT_MODEL"])

        classifier = SentimentClassifier(len(config["CLASS_NAMES"]))
        # classifier.load_state_dict(
        #     torch.load(config["PRE_TRAINED_MODEL"], map_location=self.device)
        # )
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)

    def predict(self, text):
        # max_length=config["MAX_SEQUENCE_LEN"]
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=config["MAX_SEQUENCE_LEN"],
            # add_special_tokens=True,
            # return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            # return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)       
        with torch.no_grad():
            predicted_class_1 = self.classifier(input_ids, attention_mask)[0].item()
            predicted_class_2 = self.classifier(input_ids, attention_mask)[1].item()
            # probabilities = F.softmax(self.classifier(input_ids, attention_mask), dim=1)
        # confidence, predicted_class = torch.max(probabilities, dim=1)
        # predicted_class = predicted_class.cpu().item()
        # probabilities = probabilities.flatten().cpu().numpy().tolist()
        return (
            int2Label[predicted_class_1],
            int2Label[predicted_class_2]
            # config["CLASS_NAMES"][predicted_class],
            # 'ahmed'
            # ,
            # dict(zip(config["CLASS_NAMES"], probabilities)),
        )


model = Model()


def get_model():
    return model
