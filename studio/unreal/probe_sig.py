import unreal

out = {}
for cls, ms in [
    (unreal.IKRigController, ["get_controller", "set_skeletal_mesh", "apply_auto_generated_retarget_definition", "apply_auto_fbik", "get_retarget_chains"]),
    (unreal.IKRetargeterController, ["get_controller", "set_ik_rig", "add_default_ops", "assign_ik_rig_to_all_ops", "auto_map_chains", "auto_align_all_bones", "set_preview_mesh", "run_op_initial_setup", "snap_bone_to_ground"]),
    (unreal.IKRetargetBatchOperation, ["duplicate_and_retarget", "run_batch_retarget"]),
]:
    for m in ms:
        out[f"{cls.__name__}.{m}"] = (getattr(cls, m).__doc__ or "").strip()[:500]
out["AutoMapChainType"] = [n for n in dir(unreal.AutoMapChainType) if n.isupper()]
out["RetargetSourceOrTarget"] = [n for n in dir(unreal.RetargetSourceOrTarget) if n.isupper()]
out["factories"] = [n for n in dir(unreal) if "Factory" in n and ("IK" in n or "Retarget" in n)]
result = out
