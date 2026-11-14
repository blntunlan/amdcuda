"""Independent acceptance gate for exported 4K PBR textures (CPU, not GPU)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify_folder(folder: Path):
    import numpy as np
    from PIL import Image

    material = json.loads((folder/'material.json').read_text(encoding='utf-8'))
    width, height = material['width'], material['height']
    if (width, height) != (4096,4096):
        raise ValueError(f'Not a 4K square: {width}x{height}')
    names = ('BaseColor','Normal','Roughness','Metallic','AO','ORM','Height')
    if set(material['maps']) != set(names):
        raise ValueError('Missing or unexpected PBR maps')
    mode = {k:('RGB' if k in ('BaseColor','Normal','ORM') else 'L') for k in names}
    with _ImagesContext(folder, names) as images:
        checksums = {k: hashlib.sha256() for k in names}
        max_error=0.0
        seam_max = {k:0 for k in names}
        first_row = {}
        last_row = {}
        for k,img in images.items():
            if img.format!='PNG' or img.mode!=mode[k] or img.size!=(4096,4096):
                raise ValueError(f'{k} PNG header mismatch: {img.mode} {img.size}')
        for y in range(0,height,256):
            strips = {k: np.array(images[k].crop((0,y,width,min(height,y+256))),dtype=np.uint8)
                      for k in names}
            for k,arr in strips.items():
                checksums[k].update(arr.tobytes())
                seam_max[k] = max(seam_max[k], int(np.max(np.abs(arr[:,0].astype(np.int16)-arr[:,-1].astype(np.int16)))))
                if y==0:
                    first_row[k]=arr[0].copy()
                if y+256>=height:
                    last_row[k]=arr[-1].copy()
            orm=strips['ORM']
            if not (np.array_equal(orm[:,:,0],strips['AO']) and
                    np.array_equal(orm[:,:,1],strips['Roughness']) and
                    np.array_equal(orm[:,:,2],strips['Metallic'])):
                raise ValueError('Incorrect ORM packing')
            n=strips['Normal'].astype(np.float32)*(2/255)-1
            err=float(np.max(np.abs(np.linalg.norm(n,axis=2)-1)))
            max_error=max(max_error,err)
            if err>.015:
                raise ValueError('Invalid normal vectors')
        for k in names:
            spec=material['maps'][k]
            if checksums[k].hexdigest()!=spec['sha256_raw']:
                raise ValueError(f'Pixel checksum mismatch: {k}')
            if hashlib.sha256((folder/(k+'.png')).read_bytes()).hexdigest()!=spec['sha256_png']:
                raise ValueError(f'PNG checksum mismatch: {k}')
            seam_max[k] = max(seam_max[k], int(np.max(np.abs(first_row[k].astype(np.int16)-last_row[k].astype(np.int16)))))
            if seam_max[k] > 2:
                raise ValueError(f'Visible wrap-edge discontinuity greater than 2 LSB: {k} max={seam_max[k]}')
    return {'folder':str(folder), 'size':[width,height], 'maps':7,
            'pixel_checksums_match':True, 'png_checksums_match':True,
            'orm_exact':True, 'normal_max_length_error':round(max_error,7),
            'wrap_edges_max_diff':seam_max, 'wrap_edges_pass':True,
            'gpu_compute_confirmed':False, 'status':'PASS'}


class _ImagesContext:
    def __init__(self,root,names):
        self.root,self.names=root,names
    def __enter__(self):
        from PIL import Image
        self.images={k:Image.open(self.root/(k+'.png')) for k in self.names}
        return self.images
    def __exit__(self,*args):
        for img in self.images.values():
            img.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('folders',type=Path,nargs='+')
    p.add_argument('--report',type=Path)
    args=p.parse_args()
    stages=[]
    for folder in args.folders:
        try:
            stages.append(verify_folder(folder))
        except (ValueError,OSError,KeyError,TypeError,ImportError) as exc:
            stages.append({'folder':str(folder),'status':'FAIL','error':str(exc)})
    report={'purpose':'4K PBR CPU texture asset acceptance', 'gpu_compute_confirmed':False,
            'overall':'PASS' if all(stage['status']=='PASS' for stage in stages) else 'FAIL',
            'stages':stages}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if report['overall']=='PASS' else 1

if __name__=='__main__':
    raise SystemExit(main())
