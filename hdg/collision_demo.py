import base64
import argparse
import csv
import io
import json
import random
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
from data_process.visualisation.utils.map_vis_without_lanelet import draw_map_without_lanelet
from data_process.visualisation.utils.tracks_vis import polygon_xy_from_motionstate
from data_process.visualisation.utils.dataset_types import MotionState
from .representation import LOW,HIGH


def run(root, selections, show_vehicle_ids=False, show_crop_note=False, show_crop_line=False):
    show_crop_note = show_crop_note and show_crop_line
    root=Path(root)
    out=root/'collision_demos';out.mkdir(exist_ok=True)
    demos=[]
    for selection in selections:
        path=Path(selection['path']); name=selection['name']
        e=torch.load(path,weights_only=True,map_location='cpu')
        end=min(t for t in [e['collision_time'],e.get('arrival_time'),e.get('exit_time')-1 if e.get('exit_time') is not None else None,139] if t is not None)
        ego=e['ego'];other=e['attribution']['events'][0]['other'] if e['attribution']['events'] else -1
        terminal='COLLISION' if e['collision_time']==end else 'ARRIVAL' if e.get('arrival_time')==end else 'EXIT (last in-region frame)' if e.get('exit_time') is not None else 'HORIZON'
        raw=e['tracks'].numpy().transpose(1,2,0)[:,:end+1].astype(float)
        valid=np.isfinite(raw).all(-1)&(raw>=-.1).all(-1)
        states=raw*(HIGH-LOW)+LOW
        csv_path=out/f'{name}.csv'
        with csv_path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['track_id','frame_id','timestamp_ms','agent_type','x','y','vx','vy','psi_rad','length','width'])
            for agent in range(len(states)):
                for t in np.flatnonzero(valid[agent]):
                    v=(states[agent,t,:2]-states[agent,t-1,:2])/.1 if t>0 and valid[agent,t-1] else np.zeros(2)
                    x,y,theta=states[agent,t]
                    w.writerow([agent,int(t),int(t*100),'car',x,y,*v,theta,4.5,1.8])
        fig,axes=plt.subplots(2,1,figsize=(7.6,7),gridspec_kw={'height_ratios':[1,2]},constrained_layout=True)
        for ax in axes:
            plt.sca(ax);draw_map_without_lanelet('data_process/maps/merge.osm',ax,0,0)
            if show_crop_line:
                ax.axvline(1140,color='purple',ls='--',lw=1)
            ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
        axes[0].set_xlim(995,1149);axes[0].set_ylim(915,960);axes[0].set_title('Blue:ego Orange:contact Gray:others')
        patches=[];texts=[]
        for ax in axes:
            row=[];labels=[]
            for agent in range(len(states)):
                color='#1672bc' if agent==ego else '#eb8a23' if agent==other else '#66717c'
                patch=Polygon(np.zeros((4,2)),closed=True,facecolor=color,edgecolor='black',alpha=.8,zorder=20)
                ax.add_patch(patch);row.append(patch);labels.append(ax.text(0,0,str(agent),ha='center',fontsize=8,zorder=30,clip_on=True))
            patches.append(row);texts.append(labels)
        frames=[];encoded=[]
        for t in range(end+1):
            for k,ax in enumerate(axes):
                for agent in range(len(states)):
                    patches[k][agent].set_visible(bool(valid[agent,t]));texts[k][agent].set_visible(bool(valid[agent,t]) and show_vehicle_ids)
                    if not valid[agent,t]:continue
                    ms=MotionState(t*100);ms.x,ms.y,ms.psi_rad=states[agent,t]
                    patches[k][agent].set_xy(polygon_xy_from_motionstate(ms,1.8,4.5))
                    patches[k][agent].set_edgecolor('red' if t==end and terminal=='COLLISION' and agent in [ego,other] else 'black')
                    patches[k][agent].set_linewidth(2 if t==end and terminal=='COLLISION' and agent in [ego,other] else .6)
                    texts[k][agent].set_position((ms.x,ms.y+1.8))
            center=states[ego,t]
            axes[1].set_xlim(center[0]-17,center[0]+12);axes[1].set_ylim(center[1]-7,center[1]+9)
            axes[1].set_title('Ego; purple:crop' if show_crop_note else 'Ego close-up')
            change_events=e.get('lane_change',{}).get('events',[])
            change_label=' | LANE CHANGE CONFIRMED' if any(t>=event['confirmed_frame'] for event in change_events) else ''
            fig.suptitle(f'{name}\nframe {t}/{end} | t={t*.1:.1f}s'+change_label+(' | '+terminal if t==end else ''), fontsize=10)
            fig.canvas.draw();frame=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());frames.append(frame)
            buf=io.BytesIO();frame.save(buf,format='JPEG',quality=85);encoded.append(base64.b64encode(buf.getvalue()).decode())
        frames[-1].save(out/f'{name}_terminal.png')
        frames[0].save(out/f'{name}.gif',save_all=True,append_images=frames[1:],duration=[100]*end+[1800],loop=0)
        plt.close(fig)
        demos.append(dict(name=name,frames=encoded,terminal_frame=end,terminal=terminal,ego=ego,other=other,source=str(path)))
        print(f'{name}: frames={end+1}; GIF/CSV exported',flush=True)
    template=Path(__file__).with_name('templates').joinpath('replay.html').read_text()
    (out/'index.html').write_text(template.replace('OPTIONS',''.join('<option>'+d['name']+'</option>' for d in demos)).replace('DATA',json.dumps(demos)),encoding='utf8')
    (out/'manifest.json').write_text(json.dumps([{k:v for k,v in d.items() if k!='frames'} for d in demos],indent=2)+'\n')


def render_saved(root):
    root = Path(root)
    episodes = sorted(root.glob('group_*/episode_*.pt'))
    if not episodes:
        raise FileNotFoundError('No saved episodes found in ' + str(root))
    episodes = random.Random(7).sample(episodes, min(2, len(episodes)))
    selections = [dict(path=str(path), name=path.parent.name + '_' + path.stem)
                  for path in episodes]
    out = root / 'collision_demos'
    manifest = out / 'manifest.json'
    previous = json.loads(manifest.read_text()) if manifest.exists() else []
    run(root, selections)
    selected = {item['name'] for item in selections}
    for item in previous:
        name = item['name']
        if name in selected or Path(name).name != name:
            continue
        for suffix in ['.gif', '.csv', '_terminal.png']:
            path = out / (name + suffix)
            if path.is_file():
                path.unlink()
    return root / 'collision_demos' / 'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render saved episodes')
    parser.add_argument('--results', required=True)
    args = parser.parse_args()
    print('Demos:', render_saved(args.results))
