"""Recompute frozen K0/K1 statistics from Git evidence, with no GPU/image access."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise SystemExit('Choose a new output directory: '+str(args.output))
    with tempfile.TemporaryDirectory(prefix='hrfont-e12-rebuild-') as directory:
        work=Path(directory);metrics=work/'metrics';metrics.mkdir();e12=work/'e12';e12.mkdir()
        with tarfile.open(root/'reports/k_paper_metrics_20260917/raw_scored_metrics.tar.gz') as tar:
            for m in tar:
                assert m.isfile() and Path(m.name).name==m.name and m.name.endswith('.json')
                with tar.extractfile(m) as src,(metrics/m.name).open('xb') as dst:shutil.copyfileobj(src,dst)
        for p in (root/'reports/e12c_complete_20260917').glob('*_scores.json.gz'):
            with gzip.open(p,'rb') as src,(e12/p.stem).open('xb') as dst:shutil.copyfileobj(src,dst)
        subprocess.run([sys.executable,str(root/'scripts/k_stratified_report.py'),
            '--metrics',str(metrics),'--e12',str(e12),
            '--difficulty',str(root/'reports/k_stratified_20260917'),
            '--legacy-difficulty',str(root/'reports/k_stratified_20260917/legacy_difficulty_manifest.json'),
            '--out',str(args.output.resolve())],check=True)
    print('Rebuilt '+str(args.output.resolve()/'index.html'))


if __name__=='__main__':main()
