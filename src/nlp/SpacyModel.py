# Imports
from typing import Tuple, Iterable, List
import types
import os
import spacy
import spacy.displacy
import spacy.scorer
import spacy.tokens
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from spacy.util import minibatch, compounding
from spacy.training.example import Example
from sklearn.metrics import confusion_matrix, classification_report



class SpacyModel:
    def __init__(self):
        self.model_name: str = None
        self.model: spacy.language.Language = None
        self.initialized: bool = False
        self.dataset_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nlpDataSet')
        self.test_data_filename: str = 'test.json'
        self._test_data = None
        self.train_data_filename: str = 'train.json'
        self._train_data = None
        self.validation_data_filename: str = 'validation.json'
        self._validation_data = None
        self.pipe_name: str = None

    def refresh_methods(self):
        # Rebind all methods from the class to the instance
        for attr_name in dir(self.__class__):
            attr = getattr(self.__class__, attr_name)
            if callable(attr) and not attr_name.startswith("__"):
                setattr(self, attr_name, types.MethodType(attr, self))

    @property
    def test_data_path(self):
        return os.path.join(self.dataset_path, self.test_data_filename)

    @property
    def test_data(self):
        if self._test_data is None:
            self._test_data = self.load_data(self.test_data_path)
        return self._test_data
    
    @property
    def train_data_path(self):
        return os.path.join(self.dataset_path, self.train_data_filename)

    @property
    def train_data(self):
        if self._train_data is None:
            self._train_data = self.load_data(self.train_data_path)
        return self._train_data
    
    @property
    def validation_data_path(self):
        return os.path.join(self.dataset_path, self.validation_data_filename)

    @property
    def validation_data(self):
        if self._validation_data is None:
            self._validation_data = self.load_data(self.validation_data_path)
        return self._validation_data

    def load_data(self, data_path: str):
        with open(data_path, "r") as file:
            data = json.load(file)
        prepared = self.prepare_data(data)
        return prepared

    def prepare_data(self, data):
        return data

    def get_sentence(self, data):
        return data[0]

    def disabled_pipes_names(self):
        if not self.pipe_name:
            return []
        return [pipe for pipe in self.model.pipe_names if pipe != self.pipe_name]
    
    def predict_sentence(self, sentence: str):
        doc: spacy.tokens.Doc = self.model(sentence)
        return self.on_prediction(sentence, doc)

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
            return self.model.get_pipe(pipe_name)

    def load_model(self, model_name: str):
        self.model = spacy.load(model_name)

    def ensure_initialized(self):
        if not self.initialized:
            with self.model.disable_pipes(*self.disabled_pipes_names()):
                self.model.resume_training()
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
    def __init__(self, category_names: Tuple[str, str]):
        super().__init__()
        # Only one category name is provided, so we create the second one by adding "NOT_" to the provided category name
        if isinstance(category_names, str):
            category_names = (category_names, "NOT_" + category_names)
        elif len(category_names) == 1:
            category_names = (category_names[0], "NOT_" + category_names[0])
        self.category_names = category_names

    @property
    def true_category(self):
        return self.category_names[0]
    
    @property
    def false_category(self):
        return self.category_names[1]

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
    def __init__(self, label_names: Iterable[str]):
        super().__init__()
        self.label_names = label_names

    def load_model(self, model_name: str):
        super().load_model(model_name)
        pipe = self.get_or_load_pipe("ner")

        for label in self.label_names:
            pipe.add_label(label)

    def on_prediction(self, sentence: str, doc: "spacy.tokens.Doc"):        
        return None, doc
    
    def get_expected(self, dataset_item):
        return Example.from_dict(self.model.make_doc(self.get_sentence(dataset_item)), dataset_item[1])

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
