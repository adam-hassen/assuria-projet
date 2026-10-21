# Données du projet

| Dossier | Contenu | Mise à jour |
| --- | --- | --- |
| [`objectif1-offres/`](objectif1-offres/) | Offres d'assurance auto et moto en Europe (garanties, exclusions, franchises), pour personnaliser les offres | Automatique |
| [`objectif1-simulation/`](objectif1-simulation/) | Prix de base de chaque offre, simulateur de tarif, profils générés et prix simulés | Automatique |
| [`objectif3-sinistres/`](objectif3-sinistres/) | Sinistres auto et moto en Europe, nettoyés, pour prédire le coût d'un sinistre | Automatique |

Les données sont produites par le dépôt [assuria-donnees](https://github.com/adam-hassen/assuria-donnees)
(collecte chaque semaine). Elles sont recopiées ici automatiquement toutes les heures
par le workflow `.github/workflows/sync-donnees.yml`. Ne pas les modifier à la main.

Description de chaque jeu : [offres](objectif1-offres/README.md), [simulation des prix](objectif1-simulation/README.md), [sinistres](objectif3-sinistres/README.md).
