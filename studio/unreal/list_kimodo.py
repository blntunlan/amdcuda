import unreal
eal = unreal.EditorAssetLibrary
result = {"assets": [str(p) for p in eal.list_assets("/Game/Kimodo", recursive=True)]}
