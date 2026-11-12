"""Convert locally supplied upstream Pi URDF, preserving named fixed keyframes."""
from pathlib import Path
import sys,shutil,json,hashlib,xml.etree.ElementTree as ET
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=Path(sys.argv[1]).resolve()
    dest=ROOT/'src/assets/robots/pi';dest.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source/'meshes',dest/'meshes',dirs_exist_ok=True)
    original=source/'urdf/pi_12dof_release_v1.urdf'
    tree=ET.parse(original);comp=tree.getroot().find('mujoco/compiler')
    comp.set('meshdir','meshes');comp.set('strippath','true');comp.set('fusestatic','false')
    tmp=dest/'pi_12dof_release_v1.urdf';tree.write(tmp,encoding='utf-8')
    tmp.write_text('\n'.join(line.rstrip() for line in tmp.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    spec=mujoco.MjSpec.from_file(str(tmp));fixed=spec.compile()
    base=spec.body('base_link');base.add_freejoint(name='root');base.pos=[0,0,.351]
    model=spec.compile()
    (dest/'pi.xml').write_text(spec.to_xml(),encoding='utf-8')
    # saveLastXML may serialize a machine path in compiler.meshdir; make portable.
    xml=ET.parse(dest/'pi.xml');xml.getroot().find('compiler').set('meshdir','meshes')
    motors=ET.SubElement(xml.getroot(),'actuator')
    for j in tree.getroot().findall('joint'):
        if j.get('type')=='revolute':
            effort=j.find('limit').get('effort')
            ET.SubElement(motors,'motor',name=j.get('name')+'_ctrl',joint=j.get('name'),
                          ctrllimited='true',ctrlrange=f'-{effort} {effort}')
    ET.indent(xml);xml.write(dest/'pi.xml',encoding='utf-8',xml_declaration=True)
    assert model.nq==19 and model.nv==18
    rng=np.random.default_rng(12);error=0.
    d0=mujoco.MjData(fixed);d1=mujoco.MjData(model)
    for _ in range(32):
        q=rng.uniform(fixed.jnt_range[:,0],fixed.jnt_range[:,1])
        d0.qpos[:]=q;d1.qpos[:]=np.r_[0,0,0,1,0,0,0,q]
        mujoco.mj_forward(fixed,d0);mujoco.mj_forward(model,d1)
        error=max(error,float(np.max(abs(d0.xpos-d1.xpos))))
    assert error<1e-8
    exported=mujoco.MjModel.from_xml_path(str(dest/'pi.xml'))
    assert exported.nq==19 and exported.nv==18 and exported.nu==12
    mass=float(mujoco.mj_getTotalmass(exported))
    meta=dict(source_repo='https://github.com/InternRobotics/HoST',
        source_commit='70bb580949a336a920833700e4b5dc3bf7fe87ce',
        source_urdf_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        nq=model.nq,nv=model.nv,mass=mass,fk_random_poses=32,max_fk_error=error,
        joints=[model.joint(i).name for i in range(1,model.njnt)],
        body_names=[model.body(i).name for i in range(model.nbody)],
        effort_limits={j.get('name'):float(j.find('limit').get('effort')) for j in tree.getroot().findall('joint') if j.get('type')=='revolute'},
        balanceinertia=True,source_inertia_balancing='upstream URDF already requests balanceinertia')
    (dest/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
