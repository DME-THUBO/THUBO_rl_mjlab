"""Pure I23 adapter; motor values are explicit unmeasured assumptions."""
from pathlib import Path
import json,mujoco
from mjlab.entity import EntityCfg,EntityArticulationInfoCfg
from mjlab.actuator import BuiltinPositionActuatorCfg
from .model_validation import validate
ROOT=Path(__file__).resolve().parents[1]
def settings():return json.loads((ROOT/'i23_training_settings.json').read_text())
def get_spec():
    path=ROOT/'mjcf/i23/THUBO.xml';validate(path,ROOT/'mjcf/i23/actuator_assumptions.json');spec=mujoco.MjSpec.from_file(str(path))
    for a in list(spec.actuators):spec.delete(a)
    for k in list(spec.keys):spec.delete(k)
    for g in list(spec.worldbody.geoms):
        if g.name=='floor':spec.delete(g)
    return spec
def get_robot_cfg():
    s=settings();actuator=BuiltinPositionActuatorCfg(target_names_expr=(r'^(left|right)_(hip_yaw|hip_roll|hip_pitch|knee_pitch|ankle_pitch|ankle_roll)$',),stiffness=s['stiffness_Nm_rad'],damping=s['damping_Nm_s_rad'],effort_limit=s['effort_limit_Nm'],armature=s['armature_kg_m2'],frictionloss=s['frictionloss_Nm'])
    return EntityCfg(spec_fn=get_spec,init_state=EntityCfg.InitialStateCfg(pos=tuple(s['default_root_pos_m']),joint_pos=s['default_joint_pose_rad'],joint_vel={'.*':0.0}),articulation=EntityArticulationInfoCfg(actuators=(actuator,),soft_joint_pos_limit_factor=.95))
