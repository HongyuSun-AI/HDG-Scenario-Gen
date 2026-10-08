import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from hdg.batched_loop import run_batch
from hdg.closed_loop import ReactivePlanner
from hdg.road import MergeRoad
from hdg.representation import normalize
from hdg.experiment_data import select_initializations,initialization
from hdg.night_experiment import boundary_statistics,arrays_to_tensors,episode_arrays


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(()));self.horizon=21


class RecordingDiffusion:
    def __init__(self): self.calls=[]
    def sample(self,model,shape,risk,history,mask,initial,**kw):
        self.calls.append((history.clone(),mask.clone()))

        x=history[:,:,:1].expand(shape).clone()
        x+=torch.rand(shape,generator=kw['generator'])*.0001
        return torch.where(mask.bool(),history,x)


class IndependentPlanner(ReactivePlanner):
    count=0
    def __init__(self,road):
        super().__init__(road);self.id=IndependentPlanner.count;IndependentPlanner.count+=1
    def step(self,observed,valid,speed,dt,future_env=None):
        s=observed[self.ego].copy();s[0]+=.1+self.id*.001
        return s,1.


class BatchedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.road=MergeRoad()
        self.physical=np.array([[1010.,self.road.center(1,1010.),0.],
                                [1060.,self.road.center(0,1060.),0.]])
        self.first=normalize(self.physical,np.ones(2,bool))

    def test_actual_32_batch_independent_histories_and_no_33rd_episode(self):
        IndependentPlanner.count=0;d=RecordingDiffusion();model=TinyModel()
        episodes,timing=run_batch(model,d,np.zeros((3,28,3),np.float32),self.first,[1.,0.],0,
            [1130.,self.road.center(1,1130.),0.],lambda:IndependentPlanner(self.road),horizon=21,batch_size=32)
        self.assertEqual(len(episodes),32);self.assertEqual(IndependentPlanner.count,32)
        self.assertEqual([x['size'] for x in timing['sample_batches']],[32,32])
        self.assertEqual(d.calls[0][0].shape,(32,6,21))
        self.assertFalse(torch.equal(d.calls[1][0][0,:,:11],d.calls[1][0][31,:,:11]))
        self.assertTrue(all(e['history_max_error']==0 for e in episodes))
        self.assertTrue(all(e['boundaries']==[1,11] for e in episodes))
        self.assertTrue(all(not e['preflight_only'] for e in episodes))
        for e in episodes: np.testing.assert_array_equal(e['tracks'][:,:,0].T,self.first)

    def test_initial_collision_keeps_episode_but_stops_regeneration(self):
        first=self.first.copy();first[1]=first[0];d=RecordingDiffusion()
        episodes,timing=run_batch(TinyModel(),d,np.zeros((3,28,3)),first,[1.,0.],0,
            [1130.,945.,0.],lambda:ReactivePlanner(self.road),batch_size=2,horizon=21)
        self.assertEqual(len(d.calls),1)
        self.assertTrue(all(e['collision_time']==0 for e in episodes))
        self.assertTrue(all(e['executed_frames']==21 for e in episodes))

    def test_equality_threshold_unique_source_selection_and_frame_identity(self):
        states=np.repeat(self.physical[:,None],140,axis=1);states[:,:,0]+=np.arange(140)*.4
        tracks=torch.tensor(normalize(states,np.ones((2,140),bool)).transpose(2,0,1))[None].repeat(12,1,1,1)
        data=dict(tracks=tracks,continuous_risk=torch.tensor([.9]*10+[.8]*2),
            agent_risk=torch.tensor([[.9,.2]]*12),source_indices=torch.arange(12),
            metadata={'map_path':'data_process/maps/merge.osm'})
        selected=select_initializations(data,10,3)
        self.assertEqual(len({x['source_index'] for x in selected['groups']}),10)
        self.assertTrue(all(x['dataset_index']<10 for x in selected['groups']))
        item=selected['groups'][0];_,first,_,_,_,reference=initialization(data,item)
        np.testing.assert_array_equal(first,data['tracks'][item['dataset_index'],:,:,item['frame']].T)
        np.testing.assert_array_equal(reference[:,:,0].T,first)

    def test_kcs_uses_each_episode_boundaries_and_excludes_post_collision(self):
        base=np.zeros((3,2,140),np.float32)
        episodes=[dict(tracks=base,boundaries=[1,11,21],collision_time=18,arrival_time=None),
                  dict(tracks=base,boundaries=[1,11,21,31],collision_time=None,arrival_time=30)]
        calls=[]
        def features(x,boundaries):
            calls.append(boundaries);return np.ones((3,2)),np.ones((3,2))
        with patch('hdg.night_experiment.continuity_features',side_effect=features):
            result=boundary_statistics(episodes,[base,base])
        self.assertEqual(calls,[[11],[11],[11,21],[11,21]])
        self.assertAlmostEqual(result['kcs'],1.)

    def test_episode_artifact_tensor_roundtrip(self):
        e=dict(tracks=np.zeros((3,2,140),np.float32),generation_windows=np.zeros((2,3,2,140),np.float32))
        restored=episode_arrays(arrays_to_tensors(e))
        np.testing.assert_array_equal(restored['tracks'],e['tracks'])

    def test_exact_25_epochs_best_selection_and_completed_resume(self):
        from hdg.congestion_training import train
        n=256
        data=dict(tracks=torch.rand(n,3,2,8),initial=torch.rand(n,3,28,3),risk=torch.zeros(n),
            metadata={'source_sha256':'test-source'})
        config=dict(epochs=25,feature_epochs=3,seed=7,threads=1,batch_size=256,learning_rate=1e-4,
            ema_decay=.99,eval_seed=8,eval_batch=32,evaluation_epochs=list(range(20,26)),
            temporal=dict(dim=8,depth=1,heads=2,cells=28),diffusion=dict(steps=2))
        calls=[]
        def score(*args):
            epoch=len(calls)+20;calls.append(epoch)
            return dict(fid=float(abs(epoch-22)),fid_all_reference=0.)
        with tempfile.TemporaryDirectory() as directory:
            with patch('hdg.congestion_training.frozen_features',return_value=None), \
                 patch('hdg.congestion_training.extract',return_value=np.zeros((256,4))), \
                 patch('hdg.congestion_training.real_initial_samples',return_value=torch.zeros(256,3,2,8)), \
                 patch('hdg.congestion_training.fid_report',side_effect=score),patch('builtins.print'):
                train(data,directory,config,'cpu')

                train(data,directory,config,'cpu')
            self.assertEqual(calls,list(range(20,26)))
            best=torch.load(Path(directory)/'best_temporal.pt',weights_only=True)
            last=torch.load(Path(directory)/'temporal_last.pt',weights_only=True)
            self.assertEqual(best['epoch'],22);self.assertEqual(last['epoch'],25);self.assertEqual(last['steps'],25)
            self.assertEqual(len(list(Path(directory).glob('temporal_epoch_*.pt'))),6)
            records=[json.loads(x) for x in (Path(directory)/'training.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),25);self.assertTrue(all(x['seen']==256 for x in records))
            def stop_after_26(start):
                self.assertEqual(start,26)
                yield 26
                raise KeyboardInterrupt()
            with patch('hdg.congestion_training.frozen_features',return_value=None), \
                 patch('hdg.congestion_training.itertools.count',side_effect=stop_after_26), \
                 patch('builtins.print'):
                with self.assertRaises(KeyboardInterrupt):
                    train(data,directory,dict(config,epochs=None),'cpu')
            continued=torch.load(Path(directory)/'temporal_last.pt',weights_only=True)
            self.assertEqual(continued['epoch'],26);self.assertEqual(continued['steps'],26)
            self.assertTrue((Path(directory)/'temporal_epoch_0026.pt').exists())
            self.assertEqual(len(list(Path(directory).glob('evaluation_epoch_*.json'))),6)

    def test_runner_persists_verified_episodes_and_resumes_without_sampling(self):
        from hdg.night_experiment import run_dataset
        class DescribedPlanner(ReactivePlanner):
            def describe(self): return {'name':self.name}
        states=np.repeat(self.physical[:,None],140,axis=1);states[:,:,0]+=np.arange(140)*.1
        data=dict(tracks=torch.tensor(normalize(states,np.ones((2,140),bool)).transpose(2,0,1))[None],
            continuous_risk=torch.tensor([.95]),agent_risk=torch.tensor([[.95,.2]]),source_indices=torch.tensor([0]),
            metadata={'map_path':'data_process/maps/merge.osm','source_sha256':'test'})
        model=TinyModel();model.horizon=140;model.agents=2;diffusion=RecordingDiffusion()
        config=dict(groups=1,batch_size=2,execution=10,seed=7,risk_condition=1.,cfg=1.)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'dummy.pt';checkpoint.write_bytes(b'test checkpoint hash')
            with patch('hdg.night_experiment.load_model',return_value=(model,diffusion,{})), \
                 patch('hdg.pdm_planner.PDMClosedAdapter',DescribedPlanner),patch('builtins.print'):
                report=run_dataset(data,checkpoint,Path(directory)/'run',config,device='cpu')
                before=len(diffusion.calls)
                resumed=run_dataset(data,checkpoint,Path(directory)/'run',config,device='cpu')
            self.assertEqual(report['counts']['episodes'],2)
            self.assertEqual(len(diffusion.calls),before)
            self.assertEqual(report['table2'],resumed['table2'])
            self.assertEqual(len(list((Path(directory)/'run/group_00').glob('episode_*.pt'))),2)

    def test_combined_table_pools_counts_instead_of_averaging_ratios(self):
        from hdg.night_experiment import combined_table
        moments={'real':[dict(n=3,sum=[0.,0.],outer=[[2.,0.],[0.,2.]])]*2,
                 'generated':[dict(n=3,sum=[0.,0.],outer=[[2.,0.],[0.,2.]])]*2}
        high=dict(episode_counts=[dict(collision=True,at_fault=True,verified=True)],
            continuity={'moments':moments},table2={'TTC_mean_s':0.,'TTC_std_s':0.})
        jam=dict(episode_counts=[dict(collision=False,at_fault=False,verified=False)]*3,
            continuity={'moments':moments},table2={'TTC_mean_s':2.,'TTC_std_s':1.})
        result=combined_table(high,jam)
        self.assertEqual(result['Coll_pct'],25.);self.assertEqual(result['ACR_pct'],25.)
        self.assertEqual(result['VARC_pct'],100.);self.assertAlmostEqual(result['KCS'],1.)

    def test_calibrated_sidecar_keeps_agent_risk_aligned_after_training_sort(self):
        from types import SimpleNamespace
        from hdg.experiment_data import prepare_congestion
        from hdg.relabel import sha256
        tracks=torch.full((2,3,2,140),.5);tracks[:,:,0,50:]=-1
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.pth';source.write_bytes(b'trusted mock source')
            sidecar=root/'risk.pt';map_path='data_process/maps/merge.osm'
            torch.save(dict(source_indices=torch.arange(2),agent_risk=torch.tensor([[.9,.2],[.89,.1]]),
                metadata={'source_sha256':sha256(source),'map_sha256':sha256(map_path),'parameters':{'kappa':6.8}}),sidecar)
            with patch('hdg.experiment_data.legacy_load',return_value=SimpleNamespace(data=tracks)), \
                 patch('hdg.experiment_data.assess_risk_vectorized',side_effect=AssertionError('Must reuse calibrated labels')):
                data=prepare_congestion(source,root/'ready',map_path,6.8,sidecar)
                with self.assertRaisesRegex(ValueError,'Calibrated labels mismatch'):
                    prepare_congestion(source,root/'wrong',map_path,7.,sidecar)
            torch.testing.assert_close(data['risk'],torch.tensor([1.,0.]))
            torch.testing.assert_close(data['agent_risk'],torch.tensor([[.2,.9],[.1,.89]]))


if __name__=='__main__': unittest.main()
