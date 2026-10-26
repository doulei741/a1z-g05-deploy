"""Summarize a saved run without touching devices."""
import argparse
import json
from pathlib import Path
import numpy as np


def describe(values):
    a=np.asarray(values,dtype=float)
    if not len(a):
        return None
    return {'n':len(a),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),
            'p99':float(np.percentile(a,99)),'max':float(np.max(a))}


def summarize(folder):
    rows=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    targets=[r for r in rows if r['type']=='target'];predictions=[r for r in rows if r['type']=='prediction']
    times=[r['monotonic_s'] for r in targets]
    smoothing=[r['tick_s'] for r in rows if r['type']=='smoothing_tick']
    return {'summary':json.loads((folder/'summary.json').read_text()),
            'provenance':next(r for r in rows if r['type']=='provenance'),
            'request_ms':describe([r['request_s']*1000 for r in predictions]),
            'target_interval_ms':describe(np.diff(times)*1000),
            'smoothing_interval_ms':describe(np.diff(smoothing)*1000),
            'schedule_lateness_ms':describe([(r['monotonic_s']-r['scheduled_s'])*1000 for r in targets if r.get('scheduled_s') is not None]),
            'raw_joint_delta_rad':describe([float(np.max(np.abs(np.asarray(b['target'])[[*range(6),*range(7,13)]]-np.asarray(a['target'])[[*range(6),*range(7,13)]]))) for a,b in zip(targets,targets[1:])]),
            'buffer_exhaustions':sum(r['type']=='buffer_exhausted' for r in rows),
            'physical_feedback_recorded':any(r.get('feedback') is not None for r in targets),
            'task_success_verified':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();result=summarize(a.folder);text=json.dumps(result,indent=2,ensure_ascii=False)
    if a.output:
        a.output.write_text(text+'\n')
    print(text)


if __name__=='__main__':
    main()
