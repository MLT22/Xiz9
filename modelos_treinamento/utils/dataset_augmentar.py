"""
utils/dataset_augmentar.py
─────────────────────────────────────────────────────────────────────────────
Script para adicionar imagens ao dataset Xiz9 mantendo:
  • proporção train/test do dataset original
  • sem duplicatas (verificação por hash MD5)
  • ordem aleatória (sklearn train_test_split)
  • augmentation opcional via Keras ImageDataGenerator

Dependências:
    pip install scikit-learn tensorflow pillow

Uso:
    python utils/dataset_augmentar.py

Estrutura esperada do dataset base:
    dataset/
    ├── train/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import hashlib
import shutil
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Keras é opcional — script funciona sem ele (só sem augmentation)
try:
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator, img_to_array, load_img, array_to_img
    )
    KERAS_OK = True
except ImportError:
    KERAS_OK = False

# ─── Constantes ───────────────────────────────────────────────────────────────

EXTENSOES_VALIDAS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
ARQUIVO_HASHES    = "dataset_hashes.json"   # salvo na raiz do dataset

# ─── Hashes ───────────────────────────────────────────────────────────────────

def hash_imagem(caminho: Path) -> str:
    """Gera hash MD5 do conteúdo binário da imagem."""
    h = hashlib.md5()
    with open(caminho, 'rb') as f:
        for bloco in iter(lambda: f.read(65536), b""):
            h.update(bloco)
    return h.hexdigest()


def carregar_hashes(dataset_root: Path) -> dict:
    """Carrega hashes existentes ou retorna dict vazio."""
    caminho = dataset_root / ARQUIVO_HASHES
    if caminho.exists():
        with open(caminho, 'r') as f:
            return json.load(f)
    return {}


def salvar_hashes(dataset_root: Path, hashes: dict):
    """Salva hashes no arquivo JSON."""
    caminho = dataset_root / ARQUIVO_HASHES
    with open(caminho, 'w') as f:
        json.dump(hashes, f, indent=2)
    print(f"  Hashes salvos em: {caminho.name}  ({len(hashes)} entradas)")


def construir_hashes_dataset(dataset_root: Path) -> dict:
    """
    Varre TODAS as imagens do dataset e constrói o banco de hashes.
    Chamado automaticamente quando o arquivo de hashes não existe ainda.
    """
    print("\n[Hash] Banco não encontrado — construindo pela primeira vez...")
    print("  (isso pode demorar alguns minutos dependendo do tamanho do dataset)")
    hashes = {}
    total  = 0

    for arquivo in dataset_root.rglob("*"):
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_VALIDAS:
            h = hash_imagem(arquivo)
            hashes[h] = str(arquivo.relative_to(dataset_root))
            total += 1
            if total % 500 == 0:
                print(f"  {total} imagens indexadas...")

    print(f"  Concluído: {total} imagens indexadas.")
    salvar_hashes(dataset_root, hashes)
    return hashes


# ─── Utilitários ──────────────────────────────────────────────────────────────

def listar_imagens(pasta: Path) -> list:
    """Lista todas as imagens válidas em uma pasta (não recursivo)."""
    return sorted([
        f for f in pasta.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSOES_VALIDAS
    ])


def calcular_proporcao_train(dataset_root: Path, tipo: str) -> float:
    """
    Calcula a proporção train/(train+test) para o tipo (real ou fake).
    Retorna float entre 0 e 1.
    """
    subpasta  = tipo   # "real" ou "fake"
    train_dir = dataset_root / "train" / subpasta
    test_dir  = dataset_root / "test"  / subpasta

    n_train = len(listar_imagens(train_dir)) if train_dir.exists() else 0
    n_test  = len(listar_imagens(test_dir))  if test_dir.exists()  else 0
    total   = n_train + n_test

    if total == 0:
        print(f"  [!] Nenhuma imagem encontrada para '{subpasta}'. Usando padrão 80/20.")
        return 0.8

    prop = n_train / total
    print(f"  Proporção atual ({subpasta}): {n_train} train + {n_test} test  →  {prop:.1%} / {1-prop:.1%}")
    return prop


def verificar_duplicatas(imagens: list, hashes_existentes: dict):
    """
    Separa imagens únicas de duplicatas.
    Checa duplicatas contra o dataset existente E dentro do próprio lote novo.
    Retorna (imagens_unicas, n_duplicatas, hashes_do_lote).
    """
    unicas        = []
    n_dup         = 0
    hashes_lote   = {}   # hashes só das imagens novas únicas

    for img in imagens:
        h = hash_imagem(img)
        if h in hashes_existentes:
            n_dup += 1   # duplicata com dataset existente
        elif h in hashes_lote:
            n_dup += 1   # duplicata dentro do próprio lote
        else:
            hashes_lote[h] = str(img)
            unicas.append(img)

    return unicas, n_dup


def nome_sem_conflito(pasta: Path, nome_original: str) -> Path:
    """Retorna um caminho sem conflito, adicionando sufixo aleatório se necessário."""
    dest = pasta / nome_original
    if not dest.exists():
        return dest
    stem, suf = Path(nome_original).stem, Path(nome_original).suffix
    return pasta / f"{stem}_{random.randint(10000, 99999)}{suf}"


# ─── Augmentation ─────────────────────────────────────────────────────────────

def augmentar_imagens(imagens: list, destino_tmp: Path, fator: int = 3) -> list:
    """
    Para cada imagem gera `fator` variações via Keras ImageDataGenerator.
    Salva numa pasta temporária e retorna lista de caminhos (originais + geradas).

    Parâmetros conservadores para não destruir os padrões de ruído/SRM:
      - horizontal_flip     : seguro, não afeta padrões de frequência
      - rotation_range=10   : pequena rotação
      - brightness_range    : leve variação de brilho
      - zoom_range=0.05     : zoom mínimo
      - SEM JPEG compression : evita corromper os artefatos de compressão naturais
    """
    if not KERAS_OK:
        print("  [!] Keras não disponível — augmentation pulado.")
        return imagens

    destino_tmp.mkdir(parents=True, exist_ok=True)

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=10,
        brightness_range=[0.85, 1.15],
        zoom_range=0.05,
        fill_mode='reflect',
    )

    geradas = []
    for i, img_path in enumerate(imagens, 1):
        # Copia o original
        dest_orig = destino_tmp / img_path.name
        shutil.copy2(img_path, dest_orig)
        geradas.append(dest_orig)

        # Gera variações
        try:
            img = load_img(img_path)
            arr = img_to_array(img)
            arr = arr.reshape((1,) + arr.shape)

            for j, batch in enumerate(datagen.flow(arr, batch_size=1), 1):
                nome_aug = f"{img_path.stem}_aug{j}{img_path.suffix}"
                array_to_img(batch[0]).save(destino_tmp / nome_aug)
                geradas.append(destino_tmp / nome_aug)
                if j >= fator:
                    break
        except Exception as e:
            print(f"  [!] Erro ao augmentar {img_path.name}: {e} — original mantido.")

        if i % 50 == 0:
            print(f"  Augmentation: {i}/{len(imagens)} processadas...")

    print(f"  {len(imagens)} originais  →  {len(geradas)} imagens (incluindo variações)")
    return geradas


# ─── Split e cópia ────────────────────────────────────────────────────────────

def dividir_e_copiar(
    imagens:         list,
    dataset_root:    Path,
    tipo:            str,
    prop_train:      float,
    hashes:          dict,
):
    """
    Divide as imagens aleatoriamente em train/test e copia para as pastas corretas.
    Atualiza o dicionário de hashes com cada imagem copiada.
    """
    subpasta  = tipo
    train_dir = dataset_root / "train" / subpasta
    test_dir  = dataset_root / "test"  / subpasta
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Caso especial: só 1 imagem — vai direto pro train
    if len(imagens) == 1:
        dest = nome_sem_conflito(train_dir, imagens[0].name)
        shutil.copy2(imagens[0], dest)
        hashes[hash_imagem(imagens[0])] = f"train/{subpasta}/{dest.name}"
        print(f"  1 imagem  →  train/{subpasta}/")
        return

    # sklearn train_test_split — shuffle=True e random_state=None = totalmente aleatório
    test_size   = round(1.0 - prop_train, 4)
    imgs_train, imgs_test = train_test_split(
        imagens,
        test_size=test_size,
        shuffle=True,
        random_state=None,
    )

    # Copia para train
    for img in imgs_train:
        dest = nome_sem_conflito(train_dir, img.name)
        shutil.copy2(img, dest)
        hashes[hash_imagem(img)] = f"train/{subpasta}/{dest.name}"

    # Copia para test
    for img in imgs_test:
        dest = nome_sem_conflito(test_dir, img.name)
        shutil.copy2(img, dest)
        hashes[hash_imagem(img)] = f"test/{subpasta}/{dest.name}"

    print(f"  {len(imgs_train)} imagens  →  train/{subpasta}/")
    print(f"  {len(imgs_test)} imagens  →  test/{subpasta}/")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    separador = "=" * 60

    print(separador)
    print("  Xiz9 — Gerenciador de Dataset")
    print(separador)

    # ── [1] Dataset base ──────────────────────────────────────────────────────
    print("\n[1] Caminho do DATASET BASE (pasta raiz):")
    dataset_input = input("  > ").strip().strip('"').strip("'")
    dataset_root  = Path(dataset_input)

    if not dataset_root.exists():
        print(f"\n  [ERRO] Pasta não encontrada: {dataset_root}")
        sys.exit(1)

    # Verifica estrutura mínima esperada
    pastas_esperadas = ["train/real", "train/fake", "test/real", "test/fake"]
    for pasta in pastas_esperadas:
        p = dataset_root / pasta
        if not p.exists():
            print(f"\n  [ERRO] Pasta ausente no dataset: {p}")
            print("  Estrutura esperada: train/real, train/fake, test/real, test/fake")
            sys.exit(1)

    print(f"  Dataset encontrado: {dataset_root}")

    # ── [2] Pasta de novas imagens ────────────────────────────────────────────
    print("\n[2] Caminho da PASTA COM AS NOVAS IMAGENS:")
    novas_input = input("  > ").strip().strip('"').strip("'")
    pasta_novas = Path(novas_input)

    if not pasta_novas.exists():
        print(f"\n  [ERRO] Pasta não encontrada: {pasta_novas}")
        sys.exit(1)

    # ── [3] Tipo das imagens ──────────────────────────────────────────────────
    print("\n[3] As imagens novas são REAIS ou FAKES?")
    print("  [1] Real")
    print("  [2] Fake")
    tipo_input = input("  > ").strip()

    if tipo_input == "1":
        tipo = "real"
    elif tipo_input == "2":
        tipo = "fake"
    else:
        print("\n  [ERRO] Opção inválida. Digite 1 ou 2.")
        sys.exit(1)

    print(f"  Tipo selecionado: {tipo.upper()}")

    # ── [4] Augmentation ──────────────────────────────────────────────────────
    usar_aug  = False
    fator_aug = 3

    if KERAS_OK:
        print("\n[4] Aplicar DATA AUGMENTATION nas novas imagens? (Keras)")
        print("  [1] Sim")
        print("  [2] Não")
        aug_resp = input("  > ").strip()

        if aug_resp == "1":
            usar_aug = True
            print("  Fator de augmentation — quantas variações por imagem? [padrão: 3]")
            fator_resp = input("  > ").strip()
            if fator_resp.isdigit() and int(fator_resp) > 0:
                fator_aug = int(fator_resp)
            print(f"  Fator: {fator_aug}x  (cada imagem gerará {fator_aug} variações + o original)")
    else:
        print("\n[4] Keras/TensorFlow não instalado — augmentation desabilitado.")
        print("    Para instalar: pip install tensorflow")

    # ══════════════════════════════════════════════════════════════════════════
    # PROCESSAMENTO
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{separador}")
    print("  Iniciando processamento...")
    print(separador)

    # Hashes do dataset existente
    hashes = carregar_hashes(dataset_root)
    if not hashes:
        hashes = construir_hashes_dataset(dataset_root)
    else:
        print(f"\n[Hash] Banco carregado: {len(hashes)} hashes existentes.")

    # Lista imagens novas
    imagens_novas = listar_imagens(pasta_novas)
    if not imagens_novas:
        print(f"\n  [ERRO] Nenhuma imagem encontrada em: {pasta_novas}")
        sys.exit(1)
    print(f"\n  {len(imagens_novas)} imagens encontradas na pasta de entrada.")

    # Verifica duplicatas
    print("\n[Hash] Verificando duplicatas...")
    imagens_unicas, n_dup = verificar_duplicatas(imagens_novas, hashes)

    if n_dup > 0:
        print(f"  ⚠  {n_dup} duplicata(s) ignorada(s).")
    else:
        print(f"  ✓  Nenhuma duplicata encontrada.")

    print(f"  {len(imagens_unicas)} imagens únicas para adicionar.")

    if not imagens_unicas:
        print("\n  Nada para adicionar (todas eram duplicatas). Encerrando.")
        sys.exit(0)

    # Calcula proporção train/test
    print("\n[Proporção] Calculando proporção atual do dataset...")
    prop_train = calcular_proporcao_train(dataset_root, tipo)

    # Augmentation (opcional)
    imagens_finais = imagens_unicas
    pasta_tmp = dataset_root / "_aug_tmp"

    if usar_aug:
        print(f"\n[Augmentation] Gerando variações (fator {fator_aug}x)...")
        imagens_finais = augmentar_imagens(imagens_unicas, pasta_tmp, fator=fator_aug)

    # Divide e copia
    print(f"\n[Split] Dividindo aleatoriamente e copiando para o dataset...")
    dividir_e_copiar(imagens_finais, dataset_root, tipo, prop_train, hashes)

    # Atualiza hashes
    print("\n[Hash] Atualizando banco de hashes...")
    salvar_hashes(dataset_root, hashes)

    # Remove pasta temporária do augmentation
    if pasta_tmp.exists():
        shutil.rmtree(pasta_tmp)
        print("  Pasta temporária removida.")

    # ── Resumo final ──────────────────────────────────────────────────────────
    print(f"\n{separador}")
    print("  ✓  Concluído com sucesso!")
    print(separador)
    print(f"  Tipo adicionado     : {tipo.upper()}")
    print(f"  Imagens únicas      : {len(imagens_unicas)}")
    if usar_aug:
        print(f"  Após augmentation   : {len(imagens_finais)}")
    print(f"  Proporção usada     : {prop_train:.1%} train / {1-prop_train:.1%} test")
    print(f"  Total no banco hash : {len(hashes)} entradas")
    print(separador)


if __name__ == "__main__":
    main()