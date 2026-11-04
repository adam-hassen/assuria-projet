"""
Charge les données du dépôt (data/) dans la base Postgres Neon, une table par fichier, rangées par objectif :
schémas objectif1, objectif2, objectif3. Chaque table est remplacée entièrement à chaque lancement.

Connexion : variable d'environnement NEON_DATABASE_URL (chaîne « postgresql://… » donnée par Neon).
Lancement :  NEON_DATABASE_URL=postgresql://... python scripts/charger_neon.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

D = Path("data")
# schéma, table, fichier, description
TABLES = [
    ("objectif1", "offres_europe", "objectif1/offres_europe/offres_europe.parquet", "Offres réelles d'assureurs européens (fiches IPID)"),
    ("objectif1", "fr_produits", "objectif1/offres_france_structurees/auto_products.csv", "Offres auto françaises : produits"),
    ("objectif1", "fr_types_garanties", "objectif1/offres_france_structurees/auto_guarantee_types.csv", "Dictionnaire des garanties"),
    ("objectif1", "fr_garanties_produits", "objectif1/offres_france_structurees/auto_product_guarantees.csv", "Garanties de chaque produit"),
    ("objectif1", "fr_conditions_produits", "objectif1/offres_france_structurees/auto_product_conditions.csv", "Exclusions et restrictions"),
    ("objectif1", "fr_niveaux_offres", "objectif1/offres_france_structurees/auto_offer_levels.csv", "Tiers, tiers +, tous risques par produit"),
    ("objectif1", "fr_vrais_prix", "objectif1/offres_france_structurees/auto_prices.csv", "Vrais prix relevés par profil"),
    ("objectif1", "fr_prix_a_partir_de", "objectif1/offres_france_structurees/auto_price_from.csv", "Prix d'entrée par assureur"),
    ("objectif1", "fr_stats_marche", "objectif1/offres_france_structurees/market_auto_guarantee_stats.csv", "Statistiques du marché (France Assureurs)"),
    ("objectif1", "profils", "objectif1/profils/profils.parquet", "Profils d'assurés (20 pays, permis du monde)"),
    ("objectif1", "profils_dictionnaire", "objectif1/profils/dictionnaire.csv", "Question du formulaire pour chaque champ"),
    ("objectif1", "prix_base", "objectif1/simulation/prix_base.parquet", "Prix de base par offre et formule"),
    ("objectif1", "offres_prix_profils", "objectif1/simulation/offres_prix_profils.parquet", "Offres, prix de base et prix simulé par profil"),
    ("objectif2", "documents", "objectif2/documents.csv", "Documents d'offres pour le RAG"),
    ("objectif2", "passages", "objectif2/passages.parquet", "Passages à indexer, avec offre et page"),
    ("objectif3", "sinistres_detailles", "objectif3/sinistres_detailles/sinistres_detailles.parquet", "Sinistres réels détaillés, 6 pays"),
    ("objectif3", "sinistres_detailles_dictionnaire", "objectif3/sinistres_detailles/dictionnaire.csv", "Sens de chaque colonne"),
    ("objectif3", "sinistres_europe", "objectif3/sinistres_europe/sinistres_europe.parquet", "Contrats avec sinistres, 7 jeux publics nettoyés"),
    ("objectif3", "fremtpl2_contrats", "objectif3/fremtpl2/fremtpl2_policies.parquet", "freMTPL2 : contrats"),
    ("objectif3", "fremtpl2_sinistres", "objectif3/fremtpl2/fremtpl2_claims.parquet", "freMTPL2 : sinistres individuels"),
]


def lire(chemin: Path) -> pd.DataFrame:
    if chemin.suffix == ".parquet":
        return pd.read_parquet(chemin)
    entete = chemin.read_text(encoding="utf-8-sig").split("\n", 1)[0]
    sep = ";" if ";" in entete else ","
    return pd.read_csv(chemin, sep=sep, encoding="utf-8-sig")


def preparer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object and df[c].map(lambda v: isinstance(v, (list, dict))).any():
            df[c] = df[c].map(lambda v: None if v is None else str(v))   # listes -> texte
    return df


def charger(engine, schema: str, table: str, df: pd.DataFrame, description: str) -> None:
    df.head(0).to_sql(table, engine, schema=schema, if_exists="replace", index=False)   # crée la table avec les bons types
    tampon = io.StringIO()
    df.to_csv(tampon, index=False, header=False, na_rep="\\N")
    tampon.seek(0)
    colonnes = ", ".join(f'"{c}"' for c in df.columns)
    brut = engine.raw_connection()
    try:
        with brut.cursor() as cur:
            cur.copy_expert(f'COPY "{schema}"."{table}" ({colonnes}) FROM STDIN WITH (FORMAT csv, NULL \'\\N\')', tampon)
            cur.execute(f'COMMENT ON TABLE "{schema}"."{table}" IS %s', (description,))
        brut.commit()
    finally:
        brut.close()


def main() -> int:
    url = os.environ.get("NEON_DATABASE_URL")
    if not url:
        print("NEON_DATABASE_URL manquante")
        return 1
    url = url.replace("postgres://", "postgresql://", 1).replace("postgresql://", "postgresql+psycopg2://", 1)
    engine = create_engine(url)
    with engine.begin() as cx:
        for s in ("objectif1", "objectif2", "objectif3"):
            cx.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{s}"'))
    for schema, table, fichier, description in TABLES:
        chemin = D / fichier
        if not chemin.exists():
            print(f"absent : {fichier}")
            continue
        df = preparer(lire(chemin))
        charger(engine, schema, table, df, description)
        print(f"{schema}.{table} : {len(df)} lignes")
    with engine.connect() as cx:
        taille = cx.execute(text("SELECT pg_size_pretty(pg_database_size(current_database()))")).scalar()
    print(f"Taille de la base : {taille}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
