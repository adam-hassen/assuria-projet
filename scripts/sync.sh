#!/usr/bin/env bash
# Recopie les données du dépôt assuria-donnees (chemin en argument) dans data/, rangées par objectif.
# Chaque dossier est remis exactement comme dans la source (fichiers supprimés là-bas = supprimés ici).
set -euo pipefail
SRC="${1:?chemin du dépôt assuria-donnees}"
D=data

copie() {  # copie <source> <destination> [fichiers ou dossiers à ne pas toucher...]
  local de="$1" vers="$2"; shift 2
  [ -e "$de" ] || { echo "absent : $de"; return 0; }
  mkdir -p "$vers"
  # vider la destination, sauf les éléments gardés
  find "$vers" -mindepth 1 -maxdepth 1 $(for m in "$@"; do printf -- "-not -name %s " "${m%/}"; done) -exec rm -rf {} +
  cp -r "$de". "$vers"
  for m in "$@"; do [ -e "$de${m%/}" ] && [ ! -d "$de${m%/}" ] && rm -f "$vers${m%/}"; done
  return 0
}

# Objectif 1 : personnalisation des offres
copie "$SRC/data/clean/offres/"                    "$D/objectif1/offres_europe/"
copie "$SRC/data/externe/offres_auto_france/"      "$D/objectif1/offres_france_structurees/"
copie "$SRC/data/profils/"                         "$D/objectif1/profils/"
copie "$SRC/data/clean/simulation/"                "$D/objectif1/simulation/"

# Objectif 2 : conseiller RAG sur les contrats
copie "$SRC/data/objectif2/"                       "$D/objectif2/" documents/
copie "$SRC/data/raw/offres/"                      "$D/objectif2/documents/" manifest.json

# Objectif 3 : prédiction du coût des sinistres
mkdir -p "$D/objectif3/sinistres_europe"
for f in sinistres_europe.parquet sinistres_bruts.parquet rapport_nettoyage.json README.md; do
  [ -e "$SRC/data/clean/$f" ] && cp "$SRC/data/clean/$f" "$D/objectif3/sinistres_europe/"
done
copie "$SRC/data/objectif3/"                       "$D/objectif3/sinistres_detailles/"
copie "$SRC/data/externe/fremtpl2/"                "$D/objectif3/fremtpl2/"
