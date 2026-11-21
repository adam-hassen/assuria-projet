# Réputation des assureurs

`reputation_assureurs.csv` : note des clients et nombre d'avis pour chaque assureur de `simulation/prix_base`, relevés le
21/11/2026. Sert au conseiller de l'objectif 1 pour classer les offres selon la réputation et départager des offres de
prix proches.

| Colonne | Sens |
|---|---|
| pays, assureur | comme dans `prix_base` |
| note_sur_5 | note moyenne affichée par la plateforme |
| nombre_avis | nombre d'avis derrière la note |
| plateforme | Trustpilot quand elle existe (même échelle pour tous), sinon une autre plateforme indiquée |
| source | page où la note a été relevée |
| remarque | précision (note d'une marque proche, relevé via un comparateur…) |
| date_releve | date du relevé |

Couverture : France (18 assureurs sur 21). Pas de note trouvée pour ADEP, SG (Sogessur) et Wakam (Auto DOM). Les autres
pays restent à relever. Les notes changent avec le temps : refaire le relevé avant une présentation.
