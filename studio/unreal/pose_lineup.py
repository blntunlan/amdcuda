"""Freeze source (top row) and target (bottom row) animations at the given times; set up a camera."""
import json

import unreal

with open(UH_ARG, encoding="utf-8") as fh:  # noqa: F821
    cfg = json.load(fh)
times = cfg.get("times", [0.5, 1.5, 2.2, 3.5])
rows = [(cfg["src_mesh"], cfg["src_anim"], "Hips", 0.0), (cfg["tgt_mesh"], cfg["tgt_anim"], "pelvis", 3000.0)]
sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
for a in sub.get_all_level_actors():
    if a.get_actor_label().startswith("KMD_"):
        sub.destroy_actor(a)

P = unreal.AnimPoseExtensions
opts = unreal.AnimPoseEvaluationOptions()
made = []
probes = []
for r, (mesh_p, anim_p, hip, y) in enumerate(rows):
    mesh, anim = unreal.load_asset(mesh_p), unreal.load_asset(anim_p)
    for i, t in enumerate(times):
        h = P.get_bone_pose(P.get_anim_pose_at_time(anim, t, opts), hip, unreal.AnimPoseSpaces.WORLD).translation
        loc = unreal.Vector(i * 160.0 - h.x, y - h.y, 0)
        act = sub.spawn_actor_from_class(unreal.SkeletalMeshActor, loc, unreal.Rotator(0, 0, 0))
        act.set_actor_label(f"KMD_{r}_{i}")
        comp = act.skeletal_mesh_component
        comp.set_skinned_asset_and_update(mesh)
        comp.set_animation_mode(unreal.AnimationMode.ANIMATION_SINGLE_NODE)
        data = unreal.SingleAnimationPlayData()
        data.set_editor_property("anim_to_play", anim)
        data.set_editor_property("saved_position", float(t))
        data.set_editor_property("saved_play_rate", 0.0)
        data.set_editor_property("saved_looping", False)
        data.set_editor_property("saved_playing", False)
        comp.set_editor_property("animation_data", data)
        comp.set_update_animation_in_editor(True)
        comp.override_animation_data(anim, False, False, float(t), 0.0)
        for fn in ("reregister_component", "recreate_render_state"):
            if hasattr(comp, fn):
                getattr(comp, fn)()
        hand = "RightHand" if r == 0 else "hand_r"
        probes.append({"actor": f"KMD_{r}_{i}", "t": t, "hand_r_z": round(comp.get_socket_location(hand).z, 1),
                       "expect_z": round(P.get_bone_pose(P.get_anim_pose_at_time(anim, t, opts), hand, unreal.AnimPoseSpaces.WORLD).translation.z, 1)})
        act.tags = [unreal.Name("HarnessLevel")]
        made.append(act.get_actor_label())

cams = []
for r, (_, _, _, y) in enumerate(rows):
    cam = sub.spawn_actor_from_class(unreal.CameraActor, unreal.Vector(240, y + 700, 105),
                                     unreal.Rotator(roll=0.0, pitch=-2.0, yaw=-90.0))
    cam.set_actor_label(f"KMD_Cam{r}")
    cam.tags = [unreal.Name("HarnessLevel")]
    cam.camera_component.set_editor_property("field_of_view", 55.0)
    cams.append(cam.get_actor_label())
unreal.EditorLevelLibrary.editor_invalidate_viewports()
result = {"actors": made, "cameras": cams, "probes": probes}
