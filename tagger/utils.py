import os

from typing import List, Dict
from pathlib import Path

from modules import shared, scripts

from modules.shared import models_path

default_ddp_path = Path(models_path, 'deepdanbooru')
default_onnx_path = Path(models_path, 'TaggerOnnx')

from tagger.preset import Preset
from tagger.interrogator import Interrogator, DeepDanbooruInterrogator, WaifuDiffusionInterrogator, MLDanbooruInterrogator

preset = Preset(Path(scripts.basedir(), 'presets'))

interrogators: Dict[str, Interrogator] = {
    'wd14-vit.v1': WaifuDiffusionInterrogator(
        'WD14 ViT v1',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger'),
    'wd14-vit.v2': WaifuDiffusionInterrogator(
        'WD14 ViT v2',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2',
    ),
    'wd14-convnext.v1': WaifuDiffusionInterrogator(
        'WD14 ConvNeXT v1',
        repo_id='SmilingWolf/wd-v1-4-convnext-tagger'
    ),
    'wd14-convnext.v2': WaifuDiffusionInterrogator(
        'WD14 ConvNeXT v2',
        repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2',
    ),
    'wd14-convnextv2.v1': WaifuDiffusionInterrogator(
        'WD14 ConvNeXTV2 v1',
        repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2'
    ),
    'wd14-swinv2.v2': WaifuDiffusionInterrogator(
        'WD14 SwinV2 v1',
        repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2'
    ),
    'mld-caformer.dec-5-97527': MLDanbooruInterrogator(
        'ML-Danbooru Caformer dec-5-97527',
        repo_id='deepghs/ml-danbooru-onnx',
        model_path='ml_caformer_m36_dec-5-97527.onnx'
    ),
    'mld-tresnetd.6-30000': MLDanbooruInterrogator(
        'ML-Danbooru TResNet-D 6-30000',
        repo_id='deepghs/ml-danbooru-onnx',
        model_path='TResnet-D-FLq_ema_6-30000.onnx'
    )
}


def refresh_interrogators():
    # load deepdanbooru project
    
    ddp_path = shared.cmd_opts.deepdanbooru_projects_path
    if ddp_path is None:
        ddp_path = default_ddp_path
    onnx_path = shared.cmd_opts.onnxtagger_path
    if onnx_path is None:
        onnx_path = default_onnx_path
    os.makedirs(ddp_path, exist_ok=True)
    os.makedirs(onnx_path, exist_ok=True)

    for path in os.scandir(shared.cmd_opts.deepdanbooru_projects_path):
        if not path.is_dir():
            continue

        if not Path(path, 'project.json').is_file():
            continue

        interrogators["deepdanbooru"] = DeepDanbooruInterrogator(path.name, path)


def split_str(s: str, separator=',') -> List[str]:
    return [x.strip() for x in s.split(separator) if x]
