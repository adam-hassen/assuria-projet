#!/usr/bin/env bash
# Fine-tuning LoRA du petit modèle sur Mac (Apple Silicon), avec MLX.
# À lancer depuis la racine du dépôt :  bash objectif1/agent_profil_slm/entrainer_mac.sh [modele] [iterations]
#   modele      : Qwen/Qwen2.5-1.5B-Instruct (défaut), ou Qwen/Qwen2.5-0.5B-Instruct pour un essai rapide
#   iterations  : 1500 par défaut (= 1 passage sur les 6 000 conversations, lots de 4)
set -euo pipefail

MODELE="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
ITERS="${2:-1500}"
DOSSIER="objectif1/agent_profil_slm"
NOM="$(basename "$MODELE" | tr '[:upper:]' '[:lower:]')"
SORTIE="$DOSSIER/modeles/$NOM"

python3 -m pip install --quiet "mlx-lm>=0.22"
mkdir -p "$SORTIE"

# 1. Entraînement (réglages dans lora_mac.yaml : LoRA rang 16 sur toutes les couches, perte sur le JSON seulement)
mlx_lm.lora \
  -c "$DOSSIER/lora_mac.yaml" \
  --model "$MODELE" \
  --train \
  --data "$DOSSIER/donnees" \
  --iters "$ITERS" \
  --adapter-path "$SORTIE/adapters"

# 2. Fusion du modèle de base et de l'adaptateur LoRA (modèle complet, au format Hugging Face)
mlx_lm.fuse \
  --model "$MODELE" \
  --adapter-path "$SORTIE/adapters" \
  --save-path "$SORTIE/fusionne" \
  --de-quantize

echo
echo "Modèle fine-tuné : $SORTIE/fusionne"
echo "Évaluer :          python3 $DOSSIER/evaluer.py --backend mlx --modele $SORTIE/fusionne --nom ${NOM}-finetune"
echo "Mettre dans Ollama : bash $DOSSIER/vers_ollama.sh $SORTIE/fusionne assuria-profil-${NOM}"
