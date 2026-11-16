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

# 1. Entraînement : seule la réponse (le JSON) compte dans la perte (--mask-prompt)
mlx_lm.lora \
  --model "$MODELE" \
  --train \
  --data "$DOSSIER/donnees" \
  --fine-tune-type lora \
  --num-layers 16 \
  --batch-size 4 \
  --iters "$ITERS" \
  --learning-rate 1e-4 \
  --max-seq-length 2048 \
  --mask-prompt \
  --grad-checkpoint \
  --steps-per-report 20 \
  --steps-per-eval 250 \
  --val-batches 25 \
  --save-every 250 \
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
