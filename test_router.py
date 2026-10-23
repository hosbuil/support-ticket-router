import unittest
from pathlib import Path
from router import Router, evaluate, load_data

class RouterTests(unittest.TestCase):
    def setUp(self):
        self.train, self.test = load_data(Path(__file__).with_name('tickets.csv'))
        self.model = Router(self.train)

    def test_empty_and_unknown_abstain(self):
        for text in ['', '1234', 'zebras galaxies']:
            self.assertEqual(self.model.predict(text)['label'], 'abstain')

    def test_vocabulary_uses_training_only(self):
        self.assertNotIn('saving', self.model.vocab)
        self.assertIn('saving', self.test[7]['text'])

    def test_obvious_routing(self):
        for text, expected in [('password login account', 'account'),
                               ('wifi router internet', 'network'), ('disk storage space full', 'storage')]:
            self.assertEqual(self.model.predict(text)['label'], expected)

    def test_scores_normalized(self):
        result = self.model.predict('login router storage')
        self.assertAlmostEqual(sum(result['scores'].values()), 1)

    def test_threshold_changes_coverage(self):
        self.assertEqual(self.model.predict('login', 0)['label'], 'account')
        self.assertEqual(self.model.predict('login', 1)['label'], 'abstain')

    def test_bad_threshold(self):
        for threshold in [-1, 2, float('nan')]:
            with self.assertRaises(ValueError): self.model.predict('login', threshold)

    def test_empty_training_rejected(self):
        with self.assertRaises(ValueError): Router([])

    def test_abstentions_count_as_errors(self):
        report = evaluate(self.model, self.test, 1)
        self.assertEqual(report['coverage'], 0)
        self.assertEqual(report['accuracy_abstentions_count_as_wrong'], 0)
        self.assertIsNone(report['accepted_accuracy'])

if __name__ == '__main__': unittest.main()
