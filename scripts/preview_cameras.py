# This file is part of G05_A1Z, licensed under the Galaxea Dynamics A1Z License Agreement.
# See LICENSE.txt in the project root; upstream components retain their applicable licenses.

"""Preview/check collection YAML or deployment JSON cameras; no arms or leaders."""
import argparse,threading,time,json
from pathlib import Path
import cv2,numpy as np,yaml

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--check-only',action='store_true');p.add_argument('--duration',type=float,default=10)
    a=p.parse_args()
    if not 0<a.duration<=600:p.error('duration must be within (0, 600] seconds')
    cfg=yaml.safe_load(a.config.read_text())
    if 'camera_cfg' not in cfg:
        topology=cfg.get('topology')
        if topology not in ('pc-direct','pc-direct-single','pc-direct-single-left'):
            p.error('Expected a collection YAML or PC deployment JSON')
        single=topology!='pc-direct'
        topics=({'head':cfg.get('camera_head'),
                 'wrist':cfg.get('camera_wrist',cfg.get('camera_left_wrist'))} if single else
                {'cam_high':cfg.get('camera_head'),'cam_left_wrist':cfg.get('camera_left_wrist'),
                 'cam_right_wrist':cfg.get('camera_right_wrist')})
        width,height=cfg['image_size']
        cfg={'camera_cfg':{'camera_names':list(topics),'topics':topics,
                           'width':width,'height':height},
             'collector_cfg':{'camera_freq':cfg['control_frequency']}}
    c=cfg['camera_cfg'];names=c['camera_names']
    devices=[c['topics'].get(n) for n in names]
    if any(not x or not Path(x).exists() for x in devices) or len(set(devices))!=len(devices):
        raise ValueError('Set distinct, existing camera paths first')
    fps=cfg['collector_cfg']['camera_freq'];stop=threading.Event();latest={};stats={};threads=[]
    labels={'head':'Head','wrist':'Wrist','cam_high':'Head',
            'cam_left_wrist':'Left Wrist' if len(names)==3 else 'Wrist',
            'cam_right_wrist':'Right Wrist'}
    def capture(name,device):
        cap=None;times=[];error=None
        try:
            cap=cv2.VideoCapture(device,cv2.CAP_V4L2)
            w,h=c.get('resolutions',{}).get(name,[c['width'],c['height']])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,w);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h);cap.set(cv2.CAP_PROP_FPS,fps);cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            if not cap.isOpened():raise RuntimeError('open failed')
            while not stop.is_set():
                ok,frame=cap.read()
                if not ok or frame is None:raise RuntimeError('read failed')
                if frame.shape[:2]!=(h,w):raise RuntimeError('unexpected resolution '+str(frame.shape))
                times.append(time.monotonic());latest[name]=frame
        except Exception as exc:error=str(exc);stop.set()
        finally:
            if cap is not None:cap.release()
            gaps=[y-x for x,y in zip(times,times[1:])]
            stats[name]={'frames':len(times),'fps':(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0,'max_gap_s':max(gaps,default=0),'error':error}
    try:
        for name,device in zip(names,devices):
            t=threading.Thread(target=capture,args=(name,device),daemon=True);threads.append(t);t.start()
        start=time.monotonic()
        print('Camera only; no CAN/Leader opened. Close with Q/Esc or Ctrl+C.',flush=True)
        while not stop.wait(.02):
            if a.check_only:
                if time.monotonic()-start>=a.duration:break
                continue
            tiles=[]
            for name in names:
                frame=latest.get(name)
                tile=np.zeros((240,426,3),np.uint8) if frame is None else cv2.resize(frame,(426,240))
                cv2.putText(tile,labels.get(name,name),(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2);tiles.append(tile)
            cv2.imshow('A1Z cameras',np.hstack(tiles))
            if cv2.waitKey(1)&255 in (ord('q'),27):break
    except KeyboardInterrupt:pass
    finally:
        stop.set()
        for t in threads:t.join(timeout=2)
        if not a.check_only:cv2.destroyAllWindows()
    for name in names:
        if name not in stats:stats[name]={'error':'capture did not stop within 2 seconds'}
    print(json.dumps(stats,indent=2))
    if any(s.get('error') or not s.get('frames') for s in stats.values()):raise SystemExit(1)

if __name__=='__main__':main()
