# AssurIA

Projet ENSIM : services d'assurance intelligents combinant IA générative, petits modèles de langage, Machine Learning et RAG.

Équipe : Omar El-Moujahid, Adam Hassen, Ismail Achibat. Encadrant : Meysam Shamsi.

Périmètre : assurance **auto et moto**, **Europe**.

## Objectifs et données

| Objectif | But | Données |
| --- | --- | --- |
| **1. Personnalisation des offres** | Un agent (petit LLM local + gros LLM) qui construit le profil de l'assuré par questions, puis compare et recommande les offres réelles adaptées | [`data/objectif1/`](data/README.md#objectif-1--personnaliser-les-offres-dassurance) : 104 offres réelles, vrais prix, 100 008 profils, prix simulés |
| **2. Analyse des contrats** | Un conseiller RAG qui répond aux questions sur un contrat (garanties, exclusions, franchises) en citant les passages | [`data/objectif2/`](data/objectif2/README.md) : 104 documents d'offres, 8 456 passages |
| **3. Coût des sinistres** | Comparer des modèles de Machine Learning classiques et des architectures avec LLM pour prédire le coût d'un sinistre | [`data/objectif3/`](data/README.md#objectif-3--prédire-le-coût-des-sinistres) : 269 730 sinistres réels détaillés, 6 pays |

## Organisation du dépôt

```
data/
  objectif1/   offres_europe/  offres_france_structurees/  profils/  simulation/
  objectif2/   documents/  textes/  passages.parquet  documents.csv
  objectif3/   sinistres_detailles/  sinistres_europe/  fremtpl2/
scripts/sync.sh                     recopie des données depuis assuria-donnees
.github/workflows/sync-donnees.yml  synchronisation automatique (toutes les heures)
```

Le détail de chaque jeu de données est dans [`data/README.md`](data/README.md) et dans le README de chaque dossier.

## Liens

- Dépôt de collecte et de nettoyage des données : [assuria-donnees](https://github.com/adam-hassen/assuria-donnees)
