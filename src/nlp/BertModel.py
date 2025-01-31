# Imports
from typing import Iterable, Union
import matplotlib.pyplot as plt
import numpy as np
import re
from nlp.Model import Model

import transformers
from transformers import PreTrainedTokenizerFast, PreTrainedModel, TrainingArguments, Trainer
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from datasets import Dataset

class BertModel(Model):
    def __init__(self, model_name: str = None):
        self.model: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizerFast = None
        # TODO: Remove trainer need
        self.trainer = None
        super().__init__(model_name=model_name)

        if model_name:
            self.load_model(model_name)

class BertNERModel(BertModel):
    def __init__(self, label_names: Iterable[str], model_name: str = None):
        # Example for label_names: ["O", "B-VILLE_DEPART", "I-VILLE_DEPART", "B-VILLE_DESTINATION", "I-VILLE_DESTINATION"]
        self.label_names = label_names
        self.label_map = {label: i for i, label in enumerate(label_names)}
        super().__init__(model_name=model_name)
    
    def _data_to_iob2(self, data):
        iob2_data = []

        for entry in data:
            sentence = entry["sentence"]
            entities = entry["entities"]

            if len(entities) != 2:
                continue
            # Test for overlap
            if max(entities[0]['start'], entities[1]['start']) < min(entities[0]['end'], entities[1]['end']):
                continue

            tags = []
            tokens = []
            in_entity: str = None

            for match in re.finditer(r'\S+', sentence):
                tag = 'O'
                token = match.string[match.start(): match.end()]
                tokens.append(token)

                for entity in entities:
                    start, end, label = entity["start"], entity["end"], entity["label"]

                    # Test if the token is part of the entity
                    if not (match.start() == start or in_entity == label):
                        continue

                    # Start of entity
                    if match.start() == start:
                        tag = f"B-{label}"
                    else:
                        tag = f"I-{label}"
                    
                    # This is a multi-token entity, so we need to keep track of the entity label
                    if match.end() != end:
                        in_entity = label
                    else:
                        in_entity = None
                    break

                tags.append(self.label_map[tag])

            iob2_data.append({
                "tokens": tokens,
                "ner_tags": tags
            })

        return iob2_data

    def _tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128)
        labels = []

        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    # Special token (e.g., [CLS], [SEP], padding)
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    # First subword of a word
                    label_ids.append(label[word_id])
                else:
                    # Following subwords of a word
                    label_ids.append(-100)
                previous_word_id = word_id
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def prepare_data(self, data):
        iob2_data = self._data_to_iob2(data)
        data_dict = {
            "tokens": [example["tokens"] for example in iob2_data],
            "ner_tags": [example["ner_tags"] for example in iob2_data],
        }
        dataset = Dataset.from_dict(data_dict)
        tokenized_dataset = dataset.map(self._tokenize_and_align_labels, batched=True)
        return tokenized_dataset

    def load_model(self, model_name):
        self.model_name = model_name
        self.tokenizer = transformers.CamembertTokenizerFast.from_pretrained(model_name)
        self.model = transformers.CamembertForTokenClassification.from_pretrained("camembert-base", num_labels=len(self.label_names))

    def train(self, num_epochs: int = 10, training_args: TrainingArguments = None, output_dir="./bert-ner", logs_dir="./logs"):
        if not training_args:
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                learning_rate=5e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=num_epochs,
                weight_decay=0.01,
                save_strategy="epoch",
                logging_dir=logs_dir,
                logging_steps=10,
            )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.test_data,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )
        self.trainer.train()

    def _get_true_labels_and_predictions(self, predictions, labels):
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (-100)
        true_labels = [[self.label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [[self.label_names[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
        return true_labels, true_predictions


    def _compute_metrics(self, p, with_plot: bool = False):
        predictions, labels = p
        true_labels, true_predictions = self._get_true_labels_and_predictions(predictions, labels)

        # Calculate metrics
        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions)
        report = classification_report(true_labels, true_predictions)

        print("Classification Report:")
        print(report)
        
        if with_plot:
            metrics = ["Precision", "Recall", "F1-Score"]
            scores = [precision, recall, f1]

            plt.bar(metrics, scores, color=["blue", "orange", "green"])
            plt.title("Overall NER Metrics")
            plt.ylim(0, 1)  # Scores range from 0 to 1
            plt.ylabel("Score")
            plt.show()

        return {"precision": precision, "recall": recall, "f1": f1}

    def compute_entity_metrics(self, predictions, labels):
        true_labels, true_predictions = self._get_true_labels_and_predictions(predictions, labels)

        # Initialize dictionary for per-entity metrics
        entities = [label for label in self.label_names if label != "O"]  # Skip 'O'
        entity_metrics = {}

        for entity in entities:
            # Filter true labels and predictions for the current entity
            true_entity_labels = [
                ["O" if lbl != entity else entity for lbl in seq] for seq in true_labels
            ]
            pred_entity_labels = [
                ["O" if lbl != entity else entity for lbl in seq] for seq in true_predictions
            ]

            # Calculate precision, recall, and F1 for this entity
            precision = precision_score(true_entity_labels, pred_entity_labels)
            recall = recall_score(true_entity_labels, pred_entity_labels)
            f1 = f1_score(true_entity_labels, pred_entity_labels)

            # Store metrics
            entity_metrics[entity] = {"precision": precision, "recall": recall, "f1": f1}

        # Print per-entity metrics
        for entity, metrics in entity_metrics.items():
            print(f"{entity}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")


        # Extract per-entity metrics
        entities = list(entity_metrics.keys())
        precision = [metrics["precision"] for metrics in entity_metrics.values()]
        recall = [metrics["recall"] for metrics in entity_metrics.values()]
        f1 = [metrics["f1"] for metrics in entity_metrics.values()]

        # Plot
        x = np.arange(len(entities))
        width = 0.25

        fig, ax = plt.subplots()
        ax.bar(x - width, precision, width, label='Precision')
        ax.bar(x, recall, width, label='Recall')
        ax.bar(x + width, f1, width, label='F1-Score')

        ax.set_xlabel("Entities")
        ax.set_ylabel("Scores")
        ax.set_title("Per-Entity Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(entities)
        ax.legend()

        plt.show()


    def evaluate_model(self, dataset):
        # Predictions from the model
        predictions, labels, _ = self.trainer.predict(dataset)
        self._compute_metrics((predictions, labels), with_plot=True)
        self.compute_entity_metrics(predictions, labels)
