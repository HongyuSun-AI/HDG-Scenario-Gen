import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch

from hdg.workflow import main, parser, validate


class WorkflowTests(unittest.TestCase):
    def test_only_separate_stages_are_available(self):
        action = next(action for action in parser()._actions if hasattr(action, 'choices') and isinstance(action.choices, dict))
        self.assertEqual(set(action.choices), {'prepare', 'train', 'validate'})

    def test_prepare_uses_full_source_without_training(self):
        with TemporaryDirectory() as tmp:
            with patch('hdg.workflow.prepare_congestion', return_value={'tracks': torch.zeros(3, 3, 12, 140)}) as prepare, \
                 patch('hdg.congestion_training.train') as train:
                main(['prepare', '--dataset', 'highway', '--source', 'source.pth', '--output', tmp])
            prepare.assert_called_once_with('source.pth', Path(tmp), 'data_process/maps/merge.osm', kappa=1.6)
            train.assert_not_called()

    def test_validation_passes_scene_and_batch_counts_and_writes_metrics(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            checkpoint = tmp / 'model.pt'
            checkpoint.touch()
            config = dict(groups=3, batch_size=4, seed=7, initialization='merge_intent')
            selection = dict(groups=[dict(group=i) for i in range(3)], seed=7)
            table = {k: None for k in ['KCS', 'TTC_mean_s', 'TTC_std_s', 'Coll_pct', 'ACR_pct', 'VARC_pct', 'VARAF_pct', 'VAFR_pct']}
            table['Coll_pct'] = 25.
            with patch('hdg.congestion_intent_experiment.select_initializations', return_value=selection) as choose, \
                 patch('hdg.night_experiment.run_dataset', return_value={'table2': table}) as run, \
                 patch('hdg.collision_demo.render_saved') as render, \
                 patch('hdg.congestion_training.train') as train:
                validate({}, checkpoint, tmp / 'results', config, 'cpu')
            choose.assert_called_once_with({}, 3, 7)
            self.assertEqual(run.call_args.args[3]['batch_size'], 4)
            self.assertEqual(len(json.loads((tmp / 'results/initializations.json').read_text())['groups']), 3)
            self.assertIn('| Coll_pct | 25.000000 |', (tmp / 'results/metrics.md').read_text())
            train.assert_not_called()
            render.assert_called_once_with(tmp / 'results')

    def test_missing_prepared_data_does_not_prepare_or_train(self):
        with TemporaryDirectory() as tmp:
            with patch('hdg.workflow.prepare_congestion') as prepare, patch('hdg.congestion_training.train') as train:
                with self.assertRaises(FileNotFoundError):
                    main(['train', '--dataset', 'congestion', '--output', tmp])
            prepare.assert_not_called()
            train.assert_not_called()

    def test_merge_selection_respects_requested_count_and_deduplicates(self):
        from hdg.congestion_intent_experiment import select_initializations
        namespace = 'hdg.congestion_intent_experiment.'
        items = [dict(dataset_index=i, source_index=i) for i in range(5)]
        data = {'metadata': {'map_path': 'data_process/maps/merge.osm'}}
        event = dict(completed=True, initial_lane=2)
        def initialize(data, item):
            x = .1 + item['dataset_index'] * .01
            return None, np.array([[x, .5, .5], [x+.1, .5, .5]]), None, 0, None, None
        with patch(namespace+'candidates', return_value=items), \
             patch(namespace+'initialization', side_effect=initialize), \
             patch(namespace+'lane_change', return_value=event):
            result = select_initializations(data, 3, 7)
        self.assertEqual(len(result['groups']), 3)
        self.assertEqual([item['seed'] for item in result['groups']], [1007, 2007, 3007])
        with patch(namespace+'candidates', return_value=items), \
             patch(namespace+'initialization', return_value=initialize(data, items[0])), \
             patch(namespace+'lane_change', return_value=event):
            with self.assertRaisesRegex(ValueError, 'Only 1 distinct'):
                select_initializations(data, 2, 7)


if __name__ == '__main__':
    unittest.main()
