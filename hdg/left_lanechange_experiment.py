from .representation import LOW, HIGH
from .metrics import corners
from .closed_loop import evaluation_end

def occupied_lane(state,road):
 pts=corners(state);b=road.bounds(pts[:,0])
 for lane in range(3):
  if ((pts[:,1]<=b[lane]+1e-5)&(pts[:,1]>=b[lane+1]-1e-5)).all():return lane
 return -1

def lane_change(e,road):
 end=evaluation_end(e,140);raw=e['tracks'][:,e['ego'],:end].T
 states=raw*(HIGH-LOW)+LOW
 lanes=[occupied_lane(s,road) for s in states]
 initial=lanes[0];found=[]
 for t in range(1,len(lanes)-4):
  if initial>=0 and lanes[t]>=0 and lanes[t]!=initial and all(l==lanes[t] for l in lanes[t:t+5]):
   found.append(dict(from_lane=initial,to_lane=lanes[t],first_full_entry_frame=t,confirmed_frame=t+4));break
 return dict(completed=bool(found),events=found,initial_lane=initial,criterion='Full-body lane change, 5 frames',lanes=lanes)

