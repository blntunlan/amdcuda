import unreal

rc = unreal.IKRetargeterController.get_controller(unreal.load_asset("/Game/Kimodo/Rigs/RTG_SOMA_to_Manny"))
T = unreal.RetargetSourceOrTarget.TARGET
out = {}
for m in ("get_rotation_offset_for_retarget_pose_bone", "set_rotation_offset_for_retarget_pose_bone",
          "get_current_retarget_pose_name", "get_retarget_poses"):
    out[m] = (getattr(unreal.IKRetargeterController, m).__doc__ or "").strip()[:300]
out["pose_name"] = str(rc.get_current_retarget_pose_name(T))
for b in ("foot_l", "foot_r", "ball_l", "thigh_l", "calf_l"):
    q = rc.get_rotation_offset_for_retarget_pose_bone(b, T)
    out[f"offset_{b}"] = [round(q.x, 4), round(q.y, 4), round(q.z, 4), round(q.w, 4)]
result = out
