import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alias_resolution import execution_row


class AliasResolutionTests(unittest.TestCase):
    def row(self, name, pooling, alias=None, key='same'):
        return {'recipe_id': name, 'alias_of': alias, 'effective_key': key,
                'probability_sha256': 'recorded', 'macro_f1': .52,
                'recipe': {'kind': 'meta_mix', 'round': 2,
                           'source_recipe_ids': ['P', 'Q'], 'weights': [1., 0.],
                           'pooling': pooling, 'bias_strength': .5}}

    def test_declared_geometric_alias_replays_scored_arithmetic_recipe(self):
        original = self.row('A', 'arithmetic')
        alias = self.row('B', 'geometric', 'A')
        self.assertIs(execution_row(alias, {'A': original, 'B': alias}), original)
        self.assertEqual(alias['recipe']['pooling'], 'geometric')

    def test_equal_probability_is_not_enough_to_replace_another_execution(self):
        original = self.row('A', 'arithmetic', key='different')
        alias = self.row('B', 'geometric', 'A')
        self.assertIs(execution_row(alias, {'A': original, 'B': alias}), alias)

    def test_inconsistent_recorded_alias_fails(self):
        original = self.row('A', 'arithmetic')
        alias = self.row('B', 'geometric', 'A')
        alias['probability_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            execution_row(alias, {'A': original, 'B': alias})

    def test_alias_cycle_is_rejected(self):
        first = self.row('A', 'arithmetic', 'B')
        second = self.row('B', 'geometric', 'A')
        with self.assertRaises(ValueError):
            execution_row(first, {'A': first, 'B': second})


if __name__ == '__main__':
    unittest.main()
