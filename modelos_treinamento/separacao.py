import shutil, random
from pathlib import Path

DATASET = Path('./modelos_treinamento/archive')
random.seed(42)

for classe in ['real', 'fake']:
    src   = DATASET / 'train' / classe
    dst   = DATASET / 'val'   / classe
    dst.mkdir(parents=True, exist_ok=True)

    files = list(src.glob('*'))
    val_files = random.sample(files, int(len(files) * 0.15))

    for f in val_files:
        shutil.move(str(f), str(dst / f.name))

    print(f'{classe}: {len(val_files):,} movidas para val/')