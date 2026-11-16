"""
Le profil que le petit modèle doit remplir à partir de la conversation, et la consigne qu'on lui donne.

Le même fichier sert à générer les données, à entraîner et à évaluer : tous les modèles comparés
(le modèle fine-tuné, le modèle de base, l'approche d'Ismail) reçoivent exactement la même consigne.
"""
from __future__ import annotations

# champ : (type, valeurs possibles ou description)
CHAMPS: dict[str, tuple[str, str]] = {
    "type_vehicule": ("texte", "auto | moto"),
    "age": ("entier", "âge du conducteur principal"),
    "situation_familiale": ("texte", "celibataire | en_couple | marie_pacse | veuf_divorce"),
    "profession": ("texte", "salarie | fonctionnaire | independant | profession_liberale | commercial_itinerant | "
                            "agriculteur | etudiant | retraite | sans_emploi | chauffeur_vtc"),
    "pays_residence": ("texte", "pays où il habite, en français (France, Allemagne, Italie…)"),
    "commune": ("texte", "ville où il habite"),
    "type_permis": ("texte", "B | B+formation | A | A2 | A1 | AM"),
    "annees_permis": ("entier", "années depuis l'obtention du permis"),
    "pays_permis": ("texte", "pays où le permis a été obtenu, en français"),
    "deja_assure": ("booléen", "a déjà été assuré comme conducteur principal"),
    "annees_assure_total": ("entier", "nombre d'années d'assurance"),
    "bonus_malus": ("texte", "niveau tel qu'écrit sur le relevé du pays : CRM 0.50, classe 9, SF 12, degré 6, Stufe 4, "
                             "trede 14, B3, bonus 60 %, 5 ans de no claims bonus, 3 ans sans sinistre…"),
    "sinistres_responsables_36_mois": ("entier", "sinistres responsables sur 3 ans"),
    "sinistres_non_responsables_36_mois": ("entier", "sinistres non responsables sur 3 ans"),
    "resiliation_assureur": ("texte", "non | sinistres | non_paiement | fausse_declaration"),
    "marque_modele": ("texte", "marque et modèle du véhicule"),
    "energie": ("texte", "essence | diesel | hybride | electrique"),
    "annee_mise_en_circulation": ("entier", "année de première mise en circulation"),
    "mode_acquisition": ("texte", "comptant | credit | LOA_LLD"),
    "usage": ("texte", "prive | prive_trajet_travail | professionnel | loisirs | vtc"),
    "kilometrage_annuel": ("entier", "km par an"),
    "stationnement_nuit": ("texte", "garage_ferme | parking_prive | voie_publique"),
    "conducteur_secondaire": ("texte", "aucun | conjoint | enfant_jeune_conducteur | parent"),
    "formule_souhaitee": ("texte", "tiers | tiers_plus | tous_risques"),
    "franchise_souhaitee": ("texte", "basse | standard | haute"),
    "budget_mensuel_max": ("entier", "budget maximum par mois, dans la devise dite par le client"),
    "devise_budget": ("texte", "EUR | GBP | CHF | PLN | SEK | NOK | DKK | CZK | RON"),
    "paiement": ("texte", "mensuel | annuel"),
}
NOMS = list(CHAMPS)

CONSIGNE = (
    "Tu es l'assistant d'un comparateur d'assurance auto et moto. On te donne la conversation entre le conseiller et "
    "le client. Extrais le profil du client en JSON, avec exactement ces clés, dans cet ordre :\n"
    + "\n".join(f"- {k} ({t}) : {v}" for k, (t, v) in CHAMPS.items())
    + "\nRègles : n'écris que ce que le client a dit ou ce qui s'en déduit directement (une date donne un nombre "
    "d'années, une ville donne le pays). Si une information n'a pas été donnée, mets null. "
    "Réponds uniquement avec le JSON, sur une ligne."
)


def message_utilisateur(conversation: str) -> str:
    return f"Conversation :\n{conversation}\n\nProfil JSON :"
