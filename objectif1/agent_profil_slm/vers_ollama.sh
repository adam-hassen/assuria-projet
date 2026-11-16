#!/usr/bin/env bash
# Met le modèle fine-tuné dans Ollama, pour l'utiliser comme celui d'Ismail (même API, même évaluation).
# Usage :  bash objectif1/agent_profil_slm/vers_ollama.sh <dossier_modele_fusionne> <nom_ollama>
set -euo pipefail

SOURCE="${1:?dossier du modèle fusionné}"
NOM="${2:-assuria-profil}"
TRAVAIL="$(dirname "$SOURCE")"

# Conversion en GGUF (format d'Ollama) avec llama.cpp, quantifié en q8_0 (presque sans perte, 2 fois plus petit)
if [ ! -d "$TRAVAIL/llama.cpp" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$TRAVAIL/llama.cpp"
fi
python3 -m pip install --quiet -r "$TRAVAIL/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
python3 "$TRAVAIL/llama.cpp/convert_hf_to_gguf.py" "$SOURCE" --outtype q8_0 --outfile "$TRAVAIL/$NOM.gguf"

# Le modèle lit le gabarit de conversation dans le GGUF ; température 0 pour une extraction stable
cat > "$TRAVAIL/Modelfile" <<EOF
FROM ./$NOM.gguf
PARAMETER temperature 0
PARAMETER num_ctx 4096
EOF

(cd "$TRAVAIL" && ollama create "$NOM" -f Modelfile)
echo "Prêt : ollama run $NOM"
