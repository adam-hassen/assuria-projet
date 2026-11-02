# Données du projet AssurIA

Rangées par objectif. Tout est auto et moto, en Europe.

## Objectif 1 : personnaliser les offres d'assurance

| Dossier | Contenu |
| --- | --- |
| [`objectif1/offres_europe/`](objectif1/offres_europe/) | 104 offres réelles d'assureurs dans 9 pays (garanties, exclusions, franchises), lues dans leurs fiches IPID |
| [`objectif1/offres_france_structurees/`](objectif1/offres_france_structurees/) | 13 offres françaises détaillées garantie par garantie, avec 81 **vrais prix** par profil |
| [`objectif1/profils/`](objectif1/profils/) | 100 008 profils d'assurés dans 20 pays européens, permis de 62 pays du monde, organisés comme un formulaire de devis |
| [`objectif1/simulation/`](objectif1/simulation/) | Prix de base de chaque offre et prix simulé selon le profil. Fichier unique : `offres_prix_profils.csv` (37 445 lignes) |

## Objectif 2 : conseiller RAG sur les contrats

| Dossier | Contenu |
| --- | --- |
| [`objectif2/documents/`](objectif2/documents/) | Les 104 documents originaux des offres (PDF), pour afficher la source |
| [`objectif2/textes/`](objectif2/textes/) | Le texte de chaque document, page par page |
| [`objectif2/passages.parquet`](objectif2/) | 8 456 passages prêts à indexer, chacun avec l'offre, l'assureur et la page, pour citer la clause |

## Objectif 3 : prédire le coût des sinistres

| Dossier | Contenu |
| --- | --- |
| [`objectif3/sinistres_detailles/`](objectif3/sinistres_detailles/) | 269 730 vrais sinistres dans 6 pays, avec type, circonstances, déclaration en texte libre et découpage entraînement / validation / test |
| [`objectif3/sinistres_europe/`](objectif3/sinistres_europe/) | 268 184 contrats avec sinistres, 7 jeux publics nettoyés et fusionnés (une ligne par contrat) |
| [`objectif3/fremtpl2/`](objectif3/fremtpl2/) | freMTPL2 complet (France) : contrats et sinistres individuels |

Chaque dossier a son README (contenu, colonnes, ce qui est réel et ce qui est généré).

## D'où viennent les données

Elles sont produites par le dépôt [assuria-donnees](https://github.com/adam-hassen/assuria-donnees) (collecte et
nettoyage automatiques chaque semaine), puis recopiées ici toutes les heures par
`.github/workflows/sync-donnees.yml` (script `scripts/sync.sh`). Ne pas les modifier à la main ici : les modifier
dans assuria-donnees.
