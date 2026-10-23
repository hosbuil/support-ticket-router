"""AI-generated educational text classifier; Python standard library only."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re

def tokens(text):
    return re.findall(r'[a-z]+', text.lower())

class Router:
    def __init__(self, rows):
        self.documents = Counter()
        self.counts = {}
        self.vocab = set()
        for row in rows:
            label, text = row['label'], row['text']
            if not label or not tokens(text):
                raise ValueError('Training examples need a label and text tokens.')
            self.documents[label] += 1
            self.counts.setdefault(label, Counter()).update(tokens(text))
            self.vocab.update(tokens(text))
        if len(self.documents) < 2:
            raise ValueError('At least two classes are required.')

    def predict(self, text, threshold=.7):
        if not 0 <= threshold <= 1:
            raise ValueError('Threshold must be between zero and one.')
        words = [word for word in tokens(text) if word in self.vocab]
        if not words:
            return {'label': 'abstain', 'reason': 'no_known_tokens', 'scores': {}}
        logs = {}
        for label in sorted(self.documents):
            counts = self.counts[label]
            denom = sum(counts.values()) + len(self.vocab)
            logs[label] = math.log(self.documents[label] / sum(self.documents.values())) + sum(
                math.log((counts[word] + 1) / denom) for word in words)
        maximum = max(logs.values())
        weights = {label: math.exp(score - maximum) for label, score in logs.items()}
        scores = {label: value / sum(weights.values()) for label, value in weights.items()}
        winner = max(scores, key=scores.get)
        return {'label': winner if scores[winner] >= threshold else 'abstain',
                'reason': 'scored' if scores[winner] >= threshold else 'below_threshold',
                'scores': scores, 'known_tokens': words}

def load_data(path):
    with path.open(encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    if not rows or any(set(row) != {'split', 'label', 'text'} or row['split'] not in ('train', 'test')
                       or not row['label'] or not tokens(row['text']) for row in rows):
        raise ValueError('Expected nonempty split,label,text rows with train/test splits.')
    train = [row for row in rows if row['split'] == 'train']
    test = [row for row in rows if row['split'] == 'test']
    if not train or not test:
        raise ValueError('Both training and test examples are required.')
    if {r['label'] for r in test} - {r['label'] for r in train}:
        raise ValueError('Test labels must be represented in training.')
    normalize = lambda text: ' '.join(tokens(text))
    if {normalize(r['text']) for r in train} & {normalize(r['text']) for r in test}:
        raise ValueError('Duplicate text across train and test splits.')
    return train, test

def evaluate(model, rows, threshold):
    if not rows:
        raise ValueError('Evaluation set cannot be empty.')
    confusion = {label: Counter() for label in model.documents}
    correct = accepted = 0
    for row in rows:
        prediction = model.predict(row['text'], threshold)['label']
        confusion[row['label']][prediction] += 1
        correct += prediction == row['label']
        accepted += prediction != 'abstain'
    baseline = sorted(model.documents, key=lambda label: (-model.documents[label], label))[0]
    return {'examples': len(rows), 'accuracy_abstentions_count_as_wrong': correct / len(rows),
            'coverage': accepted / len(rows),
            'accepted_accuracy': correct / accepted if accepted else None,
            'majority_baseline_accuracy': sum(r['label'] == baseline for r in rows) / len(rows),
            'confusion_matrix': confusion}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('tickets.csv'))
    parser.add_argument('--text', help='Classify a synthetic ticket; omit for held-out evaluation')
    parser.add_argument('--threshold', type=float, default=.7)
    args = parser.parse_args()
    try:
        if not 0 <= args.threshold <= 1:
            raise ValueError('Threshold must be between zero and one.')
        train, test = load_data(args.data)
        model = Router(train)
        result = model.predict(args.text, args.threshold) if args.text is not None else evaluate(model, test, args.threshold)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(2, f'Input error: {exc}\n')

if __name__ == '__main__': main()
