"""Pinned native96 clean-data and frozen-cache interfaces for H."""
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms

CODE = Path(__file__).resolve().parents[1]
ROOT = Path('/root/projects/hrfont')
VARIANT = CODE / 'code/variants/cn2west_f123_rsi/FontDiffuser'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(VARIANT))
import train as T
from scripts.hrfont_feature_cache import EsCache, EcCache, key_es, sha256_file
from scripts.v0913_clean_lib import load_donors, load_cp_group, extra_exclude_indices
from scripts.hrfont_h import HModel

CACHE = ROOT / 'artifacts/h_20260915'
PARENT = ROOT / 'runs/G-CONT-G2-8gpu-V0914-A-S3407/global_step_5000'
PARENT0 = ROOT / 'runs/G0b-F0-V0913-BS256-A-S3407/global_step_10000'


def args_for(parent=PARENT):
    args = T.get_parser().parse_args([
        '--arm', 'F2', '--rsi_source', 'delta', '--no-support', '--freeze_encoders',
        '--warm_start_from', str(parent), '--no-parity_check',
        '--data_root', str(ROOT / 'data/fontdiffuser-p253-t295-s338-cn2west-v2'),
        '--split_manifest', str(ROOT / 'manifests/split_v3_228_16_16.json'),
        '--v0913_clean_map', str(ROOT / 'manifests/v0913_clean'),
        '--es_cache_path', str(ROOT / 'artifacts/g0/es_spatial'),
        '--ec_cache_path', str(ROOT / 'artifacts/g0/ec_multiscale'),
        '--resolution', '96', '--style_image_size', '96', '--content_image_size', '96',
        '--nshot_min', '1', '--nshot_max', '8', '--seed', '3407', '--no-tc_enabled'])
    args.style_image_size = (96, 96)
    args.content_image_size = (96, 96)
    return args


def dataset(args, split):
    def native(im):
        if im.size != (96, 96):
            raise ValueError(im.size)
        return im
    trans = transforms.Compose([native, transforms.ToTensor(), transforms.Normalize([.5], [.5])])
    return T.FontDataset(args, split, [trans, trans, trans], scr=False)


def model_for(arm, device, output):
    parent = PARENT0 if arm.startswith('H-D') else PARENT
    args = args_for(parent)
    # Encoder SHA verification also checks complete frozen caches.
    T._verify_caches(args, parent)
    torch.manual_seed(3407)
    base = T.FontDiffuserModel(unet=T.build_unet(args), style_encoder=T.build_style_encoder(args),
                              content_encoder=T.build_content_encoder(args))
    T._load_parent(base, parent, output)
    base.style_encoder.requires_grad_(False).eval()
    base.content_encoder.requires_grad_(False).eval()
    model = HModel(base, arm).to(device)
    stats = torch.load(CACHE / 'teacher_stats.pt', map_location=device, weights_only=True)
    model.teacher_mean.copy_(stats['mean'])
    model.teacher_std.copy_(stats['std'])
    return model, args


class DataContext:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.es = EsCache(Path(args.es_cache_path))
        self.ec = EcCache(Path(args.ec_cache_path))
        donors = load_donors(args.v0913_clean_map)
        cp_group = load_cp_group(args.v0913_clean_map)
        fonts = donors['all']
        args._v0913_donor_exclude = lambda cp: extra_exclude_indices(fonts, cp, donors, cp_group)
        self.library = T._LibraryEs(self.es, fonts, T._style_chars_from_cache(self.es))
        man = json.loads((CACHE / 'manifest.json').read_text())
        assert man['state'] == 'COMPLETE' and man['es_sha256'] == self.es.manifest['es_checkpoint_sha256']
        assert man['keys_sha256'] == sha256_file(Path(args.es_cache_path) / 'keys.txt')
        n = len(self.es.table.keys)
        assert (CACHE / 'raw12.dat').stat().st_size == n * 144 * 256 * 2
        self.local = np.memmap(CACHE / 'raw12.dat', mode='r', dtype=np.float16, shape=(n, 144, 256))

    @torch.no_grad()
    def conditions(self, samples, cfg, source, no_delta=False):
        style, queries, _, _, _, _ = T._style_conditions(self.es, samples, self.device)
        content = T._content_features(self.ec, samples, torch.zeros_like(cfg), self.device)
        query = content[-1].clone()
        content = [x.masked_fill(cfg[:, None, None, None], 0) for x in content]
        structure = T._structure_features(self.es, self.ec, self.library, samples, queries,
                                           self.args, source | cfg, self.device)
        if no_delta:
            structure = [torch.zeros_like(x) for x in structure]
        lengths = [len(x) for x in samples['ref_chars']]
        refs = torch.zeros(len(lengths), max(lengths), 144, 256)
        keep = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
        for i, (split, font, chars) in enumerate(zip(samples['split'], samples['font_stem'], samples['ref_chars'])):
            rows = [self.es.table.index[key_es(split, font, c)] for c in chars]
            refs[i, :len(chars)] = torch.from_numpy(np.array(self.local[rows], copy=True)).float()
        return style, refs.to(self.device), query, keep.to(self.device), content, structure


def batch_to(samples, device):
    b = T.CollateFN()(samples)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}


def seed_episode(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)
