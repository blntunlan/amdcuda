import unreal

def names(cls, filt=""):
    return [n for n in dir(cls) if not n.startswith("_") and filt in n.lower()]

ar = unreal.AssetRegistryHelpers.get_asset_registry()
manny = [str(a.package_name) for a in ar.get_assets_by_path("/Game/Characters/Mannequins", recursive=True)]
result = {
    "ikrig_ctrl": names(unreal.IKRigController),
    "rtg_ctrl": names(unreal.IKRetargeterController),
    "batch": names(unreal.IKRetargetBatchOperation),
    "enums": [n for n in dir(unreal) if any(k in n for k in ("AutoMap", "RetargetSourceOrTarget", "RetargetOp"))],
    "mannequin_assets": manny,
    "interchange": hasattr(unreal, "InterchangeManager"),
}
