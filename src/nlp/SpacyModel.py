# Imports
from typing import Tuple, Iterable, Union, List
import spacy
import spacy.displacy
import spacy.scorer
import spacy.tokens
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nlp.Model import Model

from spacy.util import minibatch, compounding
from spacy.training.example import Example
from sklearn.metrics import confusion_matrix, classification_report

NlpSentence = Union[str, spacy.language.Doc]


class SpacyModel(Model):
    def __init__(self, model_name: str = None):
        self.model: spacy.language.Language = None
        self.initialized: bool = False
        self.resume_training: bool = False
        self.pipe_name: str = None
        super().__init__(model_name=model_name)

        if model_name:
            self.load_model(model_name)

    def disabled_pipes_names(self):
        if not self.pipe_name:
            return []
        return [pipe for pipe in self.model.pipe_names if pipe != self.pipe_name]

    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)
    
    def predict_sentence(self, sentence: str):
        doc: spacy.tokens.Doc = self.model(sentence)
        return self.on_prediction(sentence, doc)

    def normalize_sentence(self, sentence: NlpSentence) -> spacy.language.Doc:
        if isinstance(sentence, str):
            return self.model(sentence)
        return sentence

    def predict_dataset(self, dataset, with_expected: bool = False):
        predictions = []
        for item in dataset:
            sentence = self.get_sentence(item)
            converted, predicted = self.predict_sentence(sentence)
            if with_expected:
                expected = self.get_expected(item)
            else:
                expected = None
            predictions.append((converted, predicted, expected))
        return predictions

    def on_prediction(self, sentence: str, doc: "spacy.tokens.Doc"):
        raise NotImplementedError("on_prediction method not implemented")

    def get_expected(self, dataset_item):
        raise NotImplementedError("get_expected method not implemented")

    def get_or_load_pipe(self, pipe_name: str, set_as_pipe_name: bool = True):
        if set_as_pipe_name:
            self.pipe_name = pipe_name
        if pipe_name not in self.model.pipe_names:
            return self.model.add_pipe(pipe_name, last=True)
        else:
            if set_as_pipe_name:
                self.resume_training = True
            return self.model.get_pipe(pipe_name)

    def load_model(self, model_name: str):
        self.model = spacy.load(model_name)

    def ensure_initialized(self):
        if not self.initialized:
            with self.model.disable_pipes(*self.disabled_pipes_names()):
                if self.resume_training:
                    self.model.resume_training()
                else:
                    self.model.begin_training()
            self.initialized = True

    def train(self, iterations: int = 20, drop: float = 0.2):
        with self.model.disable_pipes(*self.disabled_pipes_names()):
            self.ensure_initialized()
            for itn in range(iterations):
                random.shuffle(self.train_data)
                losses = {}
                batches = minibatch(self.train_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    examples = [Example.from_dict(self.model.make_doc(text), ann) for text, ann in zip(texts, annotations)]
                    self.model.update(examples, drop=drop, losses=losses)
                print(f"Iteration {itn + 1}, losses : {losses}")

    def save(self, name: str):
        self.model_name = name
        self.model.to_disk(name)
    
    def evaluate_model(self, dataset):
        raise NotImplementedError("evaluate_model method not implemented")

class SpacyTextCategorizerModel(SpacyModel):
    """Categorize text into one of two categories. The categories are defined by the category_names attribute and are mutually exclusive.

    The first category is considered as the positive category, and the second one is considered as the negative category.

    See https://spacy.io/api/textcategorizer for more information on the TextCategorizer component.    
    """
    def __init__(self, category_names: Tuple[str, str], model_name: str = None):
        # Only one category name is provided, so we create the second one by adding "NOT_" to the provided category name
        if isinstance(category_names, str):
            category_names = (category_names, "NOT_" + category_names)
        elif len(category_names) == 1:
            category_names = (category_names[0], "NOT_" + category_names[0])
        self.category_names = category_names
        super().__init__(model_name=model_name)

    @property
    def true_category(self):
        return self.category_names[0]
    
    @property
    def false_category(self):
        return self.category_names[1]

    def is_true_sentence(self, sentence: NlpSentence) -> bool:
        doc = self.normalize_sentence(sentence)
        return doc.cats[self.true_category] > doc.cats[self.false_category]

    def on_prediction(self, sentence: str, doc: "spacy.tokens.Doc"):
        label = max(doc.cats, key=doc.cats.get)
        return label, doc.cats
    
    def get_expected(self, dataset_item):
        return max(dataset_item[1]['cats'], key=dataset_item[1]['cats'].get)

    def label_to_bool(self, label: str):
        return label == self.true_category

    def load_model(self, model_name: str):
        super().load_model(model_name)
        pipe = self.get_or_load_pipe("textcat")
        for category in self.category_names:
            pipe.add_label(category)

    def plot_confusion_matrix(self, y_true: List[bool], y_predicted: List[bool], model_name: str = None):
        if not model_name:
            model_name = self.model_name
        conf_matrix = confusion_matrix(y_true, y_predicted)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Real')
        plt.show()

    def evaluate_model(self, dataset, name: str = None):
        self.ensure_initialized()
        if not name:
            name = self.model_name
        predicted_items = self.predict_dataset(dataset, with_expected=True)
        print(f"Model: {name}")

        y_true = []
        y_pred = []
        for label, _, expected in predicted_items:
            y_true.append(self.label_to_bool(expected))
            y_pred.append(self.label_to_bool(label))

        
        #  Confusion matrix
        self.plot_confusion_matrix(y_true, y_pred, name)
        
        # Classification report
        class_report = classification_report(y_true, y_pred)
        print("Classification Report:\n", class_report)

class SpacyNerModel(SpacyModel):
    def __init__(self, label_names: Iterable[str], model_name: str = None):
        self.label_names = label_names
        super().__init__(model_name=model_name)

    def load_model(self, model_name: str):
        super().load_model(model_name)
        pipe = self.get_or_load_pipe("ner")

        for label in self.label_names:
            pipe.add_label(label)

    def on_prediction(self, sentence: str, doc: "spacy.tokens.Doc"):        
        return None, doc
    
    def get_expected(self, dataset_item):
        return Example.from_dict(self.model.make_doc(self.get_sentence(dataset_item)), dataset_item[1])

    def get_entity(self, doc: "spacy.tokens.Doc", label: str) -> str:
        return next((ent.text for ent in doc.ents if ent.label_ == label), None)

    def plot_score(self, scores):
        metrics = scores['ents_per_type']
        labels = list(metrics.keys())  # Entity types
        precision = [metrics[label]['p'] for label in labels]
        recall = [metrics[label]['r'] for label in labels]
        f1 = [metrics[label]['f'] for label in labels]

        # Plot metrics
        x = np.arange(len(labels))  # Label positions
        width = 0.25  # Bar width

        fig, ax = plt.subplots()
        ax.bar(x - width, precision, width, label='Precision')
        ax.bar(x, recall, width, label='Recall')
        ax.bar(x + width, f1, width, label='F1-Score')

        # Add labels and legend
        ax.set_xlabel('Entity Types')
        ax.set_ylabel('Scores')
        ax.set_title('Precision, Recall, and F1-Score by Entity Type')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.show()

    def evaluate_model(self, dataset, name: str = None):
        self.ensure_initialized()
        if not name:
            name = self.model_name
        predicted_items = self.predict_dataset(dataset, with_expected=True)
        print(f"Model: {name}")

        scorer = spacy.scorer.Scorer(self.model, default_lang="fr")
        examples = []
        for _, predicted, example in predicted_items:
            predicted: "spacy.tokens.Doc"
            example: Example
            example.predicted = predicted
            examples.append(example)
        scores = scorer.score(examples)
        self.plot_score(scores)
        return scores
