"""Compare configuration semantics to e6331e0 and check isolation/contracts."""
from pathlib import Path
import sys,json,dataclasses,inspect,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RENAMES={'state':'host_state','reset_recovery':'reset_host_recovery','proprioception':'host_proprioception',
 'grouped_reward':'host_grouped_reward','apply_assistance':'apply_host_assistance','nonfinite':'host_nonfinite','tolerance':'host_tolerance'}

def canonical(value):
    if isinstance(value,slice):return {'slice':[value.start,value.stop,value.step]}
    if dataclasses.is_dataclass(value) and not isinstance(value,type):
        return {'type':type(value).__name__,'fields':{f.name:canonical(getattr(value,f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value,dict): return {'entries':[[str(k),canonical(v)] for k,v in value.items()]}
    if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
    if isinstance(value,Path):return value.name
    if callable(value):
        name=value.__qualname__
        if getattr(value,'__module__','').endswith('HoST_getUp.mdp'):name=RENAMES.get(name,name)
        item={'callable':name}
        if inspect.isfunction(value) and value.__closure__:
            item['closure']={n:canonical(c.cell_contents) for n,c in zip(value.__code__.co_freevars,value.__closure__)}
        return item
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if hasattr(value,'tolist'):return value.tolist()
    if hasattr(value,'value'):return canonical(value.value)
    raise TypeError(type(value))

def configs():
    from dme_thubo.env_cfg import dme_thubo_standing_env_cfg
    from src.tasks.velocity.config import thubo_velocity_env_cfg
    from src.tasks.velocity_getup.config import thubo_velocity_getup_env_cfg
    from src.tasks.HoST_getUp.config import recovery_cfg
    result={}
    for play in (False,True):
        result[f'standing:{play}']=dme_thubo_standing_env_cfg(play=play)
        result[f'velocity:{play}']=thubo_velocity_env_cfg(play=play)
        result[f'velocity_getup:{play}']=thubo_velocity_getup_env_cfg(play=play)
        for robot in ('thubo','pi'):
            for mode in ('relative','absolute'):
                result[f'host:{robot}:{mode}:{play}']=recovery_cfg(robot,play=play,action_mode=mode)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--record',action='store_true');a=p.parse_args()
    snapshot=ROOT/'tests/fixtures/e6331e0_configs.json'
    cfgs=configs();current={k:canonical(v) for k,v in cfgs.items()}
    if a.record:
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_text(json.dumps(current,indent=2)+'\n');print('Recorded',len(current));return
    expected=json.loads(snapshot.read_text())
    for key in current:
        if current[key]!=expected[key]:
            (ROOT/'logs/refactor_current.json').write_text(json.dumps(current,indent=2))
            raise AssertionError('Configuration changed: '+key)
    from dme_thubo.env_cfg import dme_thubo_env_cfg
    for task in ('standing','velocity','velocity_getup','host_getup'):
        x=dme_thubo_env_cfg(task=task);y=dme_thubo_env_cfg(task=task);z=dme_thubo_env_cfg(task=task,play=True)
        assert x is not y and x.actions is not y.actions
        assert x.observations['actor'].terms is not x.observations['critic'].terms
        x.observations['actor'].terms.clear()
        assert y.observations['actor'].terms and x.observations['critic'].terms and z.observations['actor'].terms
    from mjlab.tasks.registry import list_tasks
    tasks=list_tasks()
    assert 'THUBO-Rough-Velocity' not in tasks and 'THUBO-Rough-Stand' not in tasks
    for kwargs in ({'task':'bad'},{'task':'velocity','robot':'pi'},{'terrain':'rough'},{'action_mode':'bad'}):
        try:dme_thubo_env_cfg(**kwargs)
        except ValueError:pass
        else:raise AssertionError(kwargs)
    print('PASS: 14 baseline configs, mutation isolation, enabled registry and invalid-combination contracts')

if __name__=='__main__':main()
