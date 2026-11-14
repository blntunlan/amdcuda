import unreal

sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
removed = []
for a in sub.get_all_level_actors():
    lab = a.get_actor_label()
    comp = getattr(a, "skeletal_mesh_component", None)
    asset = comp.get_skinned_asset() if comp else None
    if lab.startswith("KMD_") or (asset and "/Game/Kimodo/" in asset.get_path_name()):
        removed.append(lab)
        sub.destroy_actor(a)
saved = unreal.EditorAssetLibrary.save_directory("/Game/Kimodo", only_if_is_dirty=True, recursive=True)
result = {"removed": removed, "kimodo_saved": saved,
          "kimodo_assets": unreal.EditorAssetLibrary.list_assets("/Game/Kimodo", recursive=True)}
