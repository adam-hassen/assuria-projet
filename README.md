# AssurIA

**Services d'assurance intelligents avec IA générative, petits modèles de langage, Machine Learning et RAG.**

Projet de 5e année, ENSIM (Le Mans Université).

| | |
| --- | --- |
| **Équipe** | Adam Hassen, Omar El-Moujahid, Ismail Achibat |
| **Encadrant** | Meysam Shamsi |
| **Périmètre** | Assurance **auto et moto**, en **Europe** |

Ce dépôt est le dépôt principal du projet. Il contient toutes les données prêtes à l'emploi, rangées par objectif, et les outils qui les tiennent à jour et les chargent dans une base Postgres.

---

## Sommaire

1. [Les trois objectifs](#les-trois-objectifs)
2. [Architecture](#architecture)
3. [Organisation du dépôt](#organisation-du-dépôt)
4. [Les données](#les-données)
5. [Base de données Neon](#base-de-données-neon)
6. [Automatisation](#automatisation)
7. [Utiliser les données](#utiliser-les-données)
8. [Ce qui est réel et ce qui est généré](#ce-qui-est-réel-et-ce-qui-est-généré)
9. [Installation et configuration](#installation-et-configuration)

---

## Les trois objectifs

| # | Objectif | Ce que fait le système | Données principales |
| --- | --- | --- | --- |
| **1** | **Personnalisation des offres** | Un agent pose des questions ciblées pour construire le profil de l'assuré (petit LLM local), puis compare les offres réelles et recommande les plus adaptées (LLM plus puissant), selon les garanties, le budget et les préférences. | 104 offres réelles de 61 assureurs dans 9 pays, vrais prix relevés, 100 008 profils d'assurés, prix simulés par profil |
| **2** | **Analyse des contrats (RAG)** | Un conseiller retrouve les clauses d'un contrat (garanties, exclusions, franchises, obligations) et répond en citant le passage exact, pour limiter les hallucinations. | 104 documents d'offres en 7 langues, 8 471 passages prêts à indexer |
| **3** | **Coût des sinistres** | Comparer un modèle de Machine Learning classique et des architectures hybrides (petit LLM local + modèle de prédiction + gros LLM) pour prédire le coût d'un sinistre : précision, robustesse, calibration, latence et coût. | 269 730 sinistres réels de 6 pays, avec déclaration en texte libre et découpage entraînement / validation / test |

---

## Architecture

Le projet repose sur deux dépôts et une base de données :

- **[assuria-donnees](https://github.com/adam-hassen/assuria-donnees)** collecte et nettoie les données (le « moteur »).
- **assuria-projet** (ce dépôt) reçoit les données prêtes, rangées par objectif (la « vitrine »).
- **Neon** (Postgres) contient les mêmes données sous forme de tables, pour les interroger en SQL.

```mermaid
flowchart LR
    subgraph SRC["Sources publiques"]
        S1["Fiches IPID<br/>des assureurs européens"]
        S2["Jeux de sinistres publics<br/>freMTPL2, beMTPL97, euMTPL,<br/>Espagne, Norvège, Suède"]
        S3["Prix relevés et<br/>statistiques de marché"]
    end

    subgraph DON["assuria-donnees : collecte et nettoyage"]
        P1["pipeline-offres<br/>offres, prix de base,<br/>simulateur, profils, RAG"]
        P2["pipeline-donnees<br/>sinistres européens,<br/>sinistres détaillés"]
    end

    subgraph PROJ["assuria-projet : données prêtes"]
        O1["data/objectif1"]
        O2["data/objectif2"]
        O3["data/objectif3"]
    end

    DB[("Neon Postgres<br/>schémas objectif1,<br/>objectif2, objectif3")]

    subgraph USE["Utilisation"]
        U1["Agent de recommandation"]
        U2["Conseiller RAG"]
        U3["Benchmark ML / LLM"]
    end

    S1 --> P1
    S3 --> P1
    S2 --> P2
    P1 -->|"sync toutes les heures"| O1
    P1 --> O2
    P2 --> O3
    O1 & O2 & O3 -->|"charger-neon"| DB
    O1 --> U1
    O2 --> U2
    O3 --> U3
    DB --> U1 & U2 & U3
```

### Comment les services s'enchaînent

```mermaid
flowchart TB
    subgraph A1["Objectif 1 : recommander une offre"]
        Q["Questions à l'assuré<br/>(petit LLM local)"] --> PR["Profil structuré<br/>(âge, permis, véhicule, zone, budget…)"]
        PR --> SIM["Prix estimé de chaque offre<br/>(simulateur)"]
        SIM --> REC["Comparaison et recommandation<br/>(gros LLM)"]
    end
    subgraph A2["Objectif 2 : répondre sur un contrat"]
        QU["Question de l'assuré"] --> RET["Recherche des passages proches<br/>(embeddings)"]
        RET --> REP["Réponse + citation<br/>(offre, page)"]
    end
    subgraph A3["Objectif 3 : prédire le coût"]
        DEC["Déclaration du sinistre"] --> EXT["Extraction des variables<br/>(petit LLM local)"]
        EXT --> MOD["Modèle de prédiction<br/>(CatBoost, GLM…)"]
        MOD --> EXP["Explication<br/>(gros LLM)"]
    end
```

---

## Organisation du dépôt

```
assuria-projet/
├── data/
│   ├── README.md                         index de toutes les données
│   ├── objectif1/                        personnalisation des offres
│   │   ├── offres_europe/                offres réelles lues dans les fiches IPID
│   │   ├── offres_france_structurees/    13 offres françaises détaillées + vrais prix par profil
│   │   ├── profils/                      100 008 profils d'assurés (formulaire de devis)
│   │   └── simulation/                   prix de base et prix simulés par profil
│   ├── objectif2/                        conseiller RAG
│   │   ├── documents/                    documents originaux (PDF)
│   │   ├── textes/                       texte de chaque document, page par page
│   │   ├── passages.parquet / .csv       passages à indexer, avec offre et page
│   │   └── documents.csv                 un document par ligne
│   └── objectif3/                        coût des sinistres
│       ├── sinistres_detailles/          sinistres réels + circonstances + déclaration
│       ├── sinistres_europe/             7 jeux publics nettoyés et fusionnés
│       └── fremtpl2/                     freMTPL2 complet (contrats et sinistres)
├── scripts/
│   ├── sync.sh                           recopie les données depuis assuria-donnees
│   └── charger_neon.py                   charge les données dans Neon
└── .github/workflows/
    ├── sync-donnees.yml                  synchronisation (toutes les heures)
    └── charger-neon.yml                  chargement dans Neon
```

Chaque dossier de `data/` a son propre README : contenu, colonnes, sources, limites.

---

## Les données

### Objectif 1 : personnalisation des offres

| Jeu | Contenu | Taille |
| --- | --- | --- |
| [`offres_europe`](data/objectif1/offres_europe/) | Offres réelles auto et moto : assureur, pays, garanties, exclusions, franchises, lien vers le document | 104 offres, 61 assureurs, 9 pays |
| [`offres_france_structurees`](data/objectif1/offres_france_structurees/) | 13 offres françaises détaillées garantie par garantie, conditions, niveaux (tiers, tous risques…), **81 vrais prix par profil**, statistiques de marché France Assureurs | 8 tables |
| [`profils`](data/objectif1/profils/) | Profils d'assurés organisés comme un formulaire de devis : conducteur, permis, antécédents, véhicule, usage, stationnement, couverture souhaitée | 100 008 profils, 20 pays, permis de 62 pays, 68 champs |
| [`simulation`](data/objectif1/simulation/) | Prix de base de chaque offre et prix estimé pour chaque profil, avec le détail des coefficients. Fichier unique : `offres_prix_profils.csv` | 37 445 devis |

Pays des offres : Allemagne, Autriche, Belgique, Espagne, France, Irlande, Italie, Pays-Bas, Portugal.

### Objectif 2 : conseiller RAG

| Jeu | Contenu | Taille |
| --- | --- | --- |
| [`documents/`](data/objectif2/) | Documents d'information des offres (format européen IPID), en version originale | 104 documents, 7 langues |
| [`textes/`](data/objectif2/) | Texte extrait de chaque document, page par page | 1 fichier par document |
| [`passages`](data/objectif2/) | Passages d'environ 800 caractères coupés aux fins de phrase, avec `id_offre`, `assureur`, `pays`, `langue`, `page` | 8 471 passages |

### Objectif 3 : coût des sinistres

| Jeu | Contenu | Taille |
| --- | --- | --- |
| [`sinistres_detailles`](data/objectif3/sinistres_detailles/) | Un sinistre par ligne : profil réel du contrat, **coût réel**, type de garantie, circonstances, lieu, moment, météo, blessés et **déclaration en texte libre**. Colonne `jeu` pour le découpage fixe 70 / 15 / 15 | 269 730 sinistres, 6 pays, 41 colonnes |
| [`sinistres_europe`](data/objectif3/sinistres_europe/) | 7 jeux publics de sinistres nettoyés et fusionnés, une ligne par contrat | 268 184 contrats |
| [`fremtpl2`](data/objectif3/fremtpl2/) | Jeu de référence français freMTPL2, contrats et sinistres individuels | 678 013 contrats, 26 444 sinistres |

Pays des sinistres : Italie, France, Belgique, Espagne, Norvège, Suède.

---

## Base de données Neon

Toutes les données sont aussi chargées dans une base **Postgres Neon** (région Frankfurt), une table par fichier, rangées dans trois schémas.

| Schéma | Tables |
| --- | --- |
| `objectif1` | `offres_europe`, `fr_produits`, `fr_types_garanties`, `fr_garanties_produits`, `fr_conditions_produits`, `fr_niveaux_offres`, `fr_vrais_prix`, `fr_prix_a_partir_de`, `fr_stats_marche`, `profils`, `profils_dictionnaire`, `prix_base`, `offres_prix_profils` |
| `objectif2` | `documents`, `passages` |
| `objectif3` | `sinistres_detailles`, `sinistres_detailles_dictionnaire`, `sinistres_europe`, `fremtpl2_contrats`, `fremtpl2_sinistres` |

20 tables, environ 400 Mo. Chaque table porte un commentaire qui décrit son contenu.

Pour y accéder, demander une invitation au projet Neon **AssurIA**, puis récupérer sa propre chaîne de connexion avec le bouton **Connect**.

---

## Automatisation

Tout se met à jour sans intervention :

| Quand | Où | Quoi |
| --- | --- | --- |
| Chaque lundi, 5h UTC | assuria-donnees | `pipeline-donnees` : collecte et nettoyage des sinistres, régénération des sinistres détaillés |
| Chaque lundi, 5h30 UTC | assuria-donnees | `pipeline-offres` : offres, prix de base, simulateur, profils, documents RAG |
| Toutes les heures | assuria-projet | `sync-donnees` : recopie les données dans `data/`, rangées par objectif, et commit seulement s'il y a du nouveau |
| Après chaque changement de `data/` | assuria-projet | `charger-neon` : recharge toutes les tables dans Neon |

`charger-neon` peut aussi être lancé à la main : onglet **Actions**, **Chargement dans Neon**, **Run workflow**.

> Les données de `data/` ne se modifient pas à la main dans ce dépôt : elles seraient écrasées à la prochaine synchronisation. Les changer dans assuria-donnees.

---

## Utiliser les données

### En Python

```python
import pandas as pd

# Objectif 1 : profils et prix simulés
profils = pd.read_parquet("data/objectif1/profils/profils.parquet")
devis = pd.read_parquet("data/objectif1/simulation/offres_prix_profils.parquet")

# Objectif 2 : passages à indexer pour le RAG
passages = pd.read_parquet("data/objectif2/passages.parquet")

# Objectif 3 : jeu d'entraînement et de test fixes
sinistres = pd.read_parquet("data/objectif3/sinistres_detailles/sinistres_detailles.parquet")
train = sinistres[sinistres.jeu == "entrainement"]
test = sinistres[sinistres.jeu == "test"]
```

### En SQL (Neon)

```sql
-- Nombre de sinistres et coût médian par pays
SELECT pays, count(*) AS sinistres,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY cout_sinistre_eur) AS cout_median
FROM objectif3.sinistres_detailles
GROUP BY pays ORDER BY sinistres DESC;

-- Les 5 offres les moins chères pour un profil donné
SELECT assureur, produit, formule, prix_annuel_estime
FROM objectif1.offres_prix_profils
WHERE id_profil = 'P00000' AND statut = 'ok'
ORDER BY prix_annuel_estime LIMIT 5;

-- Passages d'une offre qui parlent d'exclusions
SELECT page, texte FROM objectif2.passages
WHERE assureur ILIKE '%maaf%' AND texte ILIKE '%exclu%';
```

Formats : `.parquet` pour le code (rapide, types conservés), `.csv` pour Excel (séparateur `;`, encodage UTF-8).

---

## Ce qui est réel et ce qui est généré

Il est important de le dire clairement dans le rapport :

| Donnée | Origine |
| --- | --- |
| Offres, garanties, exclusions, franchises | **Réelles**, lues dans les documents officiels des assureurs |
| Prix relevés (81 prix par profil, prix « à partir de ») | **Réels**, relevés sur les sites des assureurs et comparateurs |
| Prix simulés par profil | **Estimés** : prix de base × coefficients appris sur freMTPL2 (fréquence × coût moyen) et règles officielles (bonus-malus, jeune conducteur) |
| Profils d'assurés | **Générés**, fictifs mais cohérents (âge minimal du permis, bonus-malus de chaque pays, villes réelles), avec une graine fixe |
| Coût des sinistres et profil du contrat | **Réels**, issus de jeux publics |
| Type, circonstances, lieu, météo et déclaration des sinistres | **Générés**, cohérents avec le coût réel et les garanties couvertes par chaque source |

Les résultats des LLM sur les déclarations mesurent donc leur capacité à extraire l'information d'un texte, pas leur performance sur de vraies déclarations.

---

## Installation et configuration

```bash
git clone https://github.com/adam-hassen/assuria-projet.git
cd assuria-projet
pip install pandas pyarrow
```

Pour charger les données dans une autre base Postgres :

```bash
pip install sqlalchemy psycopg2-binary
NEON_DATABASE_URL="postgresql://utilisateur:motdepasse@hote/base?sslmode=require" python scripts/charger_neon.py
```

Secret GitHub nécessaire (Settings, Secrets and variables, Actions) :

| Secret | Rôle |
| --- | --- |
| `NEON_DATABASE_URL` | Chaîne de connexion à la base Neon, utilisée par `charger-neon` |

---

## Liens

- Collecte et nettoyage des données : [assuria-donnees](https://github.com/adam-hassen/assuria-donnees)
- Index détaillé des données : [`data/README.md`](data/README.md)
- Benchmark de référence cité dans le sujet : [Allstate Claims Severity (Kaggle)](https://www.kaggle.com/competitions/allstate-claims-severity/data)
