"""
Génère les conversations d'entraînement du petit modèle à partir des profils d'assurés (data/objectif1/profils).

Pour chaque profil : une conversation conseiller ↔ client en français, comme lors d'un devis, et le profil JSON
que le modèle doit en extraire. Le profil attendu ne contient QUE ce que le client a dit dans la conversation :
une information jamais donnée reste null, ce qui apprend au modèle à ne rien inventer.

Variété introduite pour que le modèle généralise :
  - plusieurs formulations par question et par réponse, registre poli ou familier ;
  - le client donne parfois plusieurs informations d'un coup dès le début ;
  - réponses indirectes à convertir (« depuis 2015 », « bonus de 50 % », « dans la rue ») ;
  - corrections (« ah non pardon, 12 000 »), « je ne sais pas », questions sautées ;
  - conversations coupées en cours de route (profil partiel).

Sorties (donnees/) :
  train.jsonl, valid.jsonl, test.jsonl : format chat {"messages": [system, user, assistant]} (mlx-lm, transformers)
  test_conversations.jsonl : le jeu de test avec le détail des tours et le profil attendu, pour l'évaluation
Découpage par profil (un profil n'est que dans un seul jeu). Graine fixe.

Lancement (depuis la racine du dépôt) :  python objectif1/agent_profil_slm/generer_conversations.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from schema import CONSIGNE, NOMS, message_utilisateur  # noqa: E402

PROFILS = Path("data/objectif1/profils/profils.parquet")
SORTIE = Path(__file__).parent / "donnees"
GRAINE = 2026
ANNEE = 2026
N_TRAIN, N_VALID, N_TEST = 6000, 500, 500

R = random.Random(GRAINE)
un = R.choice


def chance(p: float) -> bool:
    return R.random() < p


PREP_PAYS = {"France": "en France", "Allemagne": "en Allemagne", "Italie": "en Italie", "Espagne": "en Espagne",
             "Royaume-Uni": "au Royaume-Uni", "Pays-Bas": "aux Pays-Bas", "Belgique": "en Belgique", "Pologne": "en Pologne",
             "Portugal": "au Portugal", "Autriche": "en Autriche", "Suisse": "en Suisse", "Suède": "en Suède",
             "Irlande": "en Irlande", "Grèce": "en Grèce", "Roumanie": "en Roumanie", "Tchéquie": "en Tchéquie",
             "Norvège": "en Norvège", "Danemark": "au Danemark", "Finlande": "en Finlande", "Luxembourg": "au Luxembourg"}


def prep_pays(nom: str) -> str:
    if nom in PREP_PAYS:
        return PREP_PAYS[nom]
    if nom.startswith(("États-Unis", "Pays-")):
        return "aux " + nom
    masculins = ("Maroc", "Brésil", "Canada", "Mexique", "Japon", "Liban", "Sénégal", "Cameroun", "Nigeria", "Pakistan",
                 "Bangladesh", "Viêt Nam", "Pérou", "Venezuela", "Royaume-Uni")
    if nom in masculins:
        return "au " + nom
    if nom == "Philippines":
        return "aux Philippines"
    return "en " + nom


DEVISES = {"EUR": ["euros", "€", "euros"], "GBP": ["livres", "£", "pounds"], "CHF": ["francs suisses", "CHF", "francs"],
           "PLN": ["zlotys", "PLN", "zlotys"], "SEK": ["couronnes", "SEK", "couronnes suédoises"],
           "NOK": ["couronnes", "NOK", "couronnes norvégiennes"], "DKK": ["couronnes", "DKK", "couronnes danoises"],
           "CZK": ["couronnes", "CZK", "couronnes tchèques"], "RON": ["lei", "RON", "lei"]}


def nombre(n: int) -> str:
    """12000 -> « 12 000 » ou « 12000 » ou « 12k »."""
    if n >= 1000 and chance(0.25) and n % 1000 == 0:
        return f"{n // 1000}k"
    return f"{n:,}".replace(",", " ") if chance(0.6) else str(n)


def ans(n: int) -> str:
    return f"{n} an" if n <= 1 else f"{n} ans"


# --------------------------------------------------------------------------- expressions du client
SITUATION = {"celibataire": ["célibataire", "je suis célibataire", "seul", "pas en couple"],
             "en_couple": ["en couple", "je vis avec ma copine", "je vis avec mon copain", "concubinage", "en couple, pas mariés"],
             "marie_pacse": ["marié", "mariée", "pacsé", "je suis marié", "pacsée avec mon compagnon"],
             "veuf_divorce": ["divorcé", "divorcée", "veuf", "séparé, divorcé"]}
PROFESSION = {"salarie": ["salarié", "je suis salarié en CDI", "employé de bureau", "je bosse dans une entreprise, en CDI",
                          "technicien, salarié"],
              "fonctionnaire": ["fonctionnaire", "prof dans un collège", "je travaille à la mairie", "infirmière à l'hôpital public",
                                "agent de la fonction publique"],
              "independant": ["indépendant", "artisan", "auto-entrepreneur", "je suis à mon compte", "plombier, à mon compte"],
              "profession_liberale": ["médecin", "avocat", "kiné en libéral", "architecte, profession libérale", "dentiste"],
              "commercial_itinerant": ["commercial, je fais beaucoup de route", "commercial itinérant", "VRP, toujours sur la route"],
              "agriculteur": ["agriculteur", "je suis exploitant agricole", "éleveur"],
              "etudiant": ["étudiant", "étudiante", "je suis en master", "je fais mes études", "étudiant en BTS"],
              "retraite": ["retraité", "à la retraite", "retraitée depuis peu"],
              "sans_emploi": ["sans emploi", "au chômage", "je cherche du travail en ce moment", "demandeur d'emploi"],
              "chauffeur_vtc": ["chauffeur VTC", "je fais du VTC"]}
ENERGIE = {"essence": ["essence", "à l'essence", "il roule à l'essence", "sans plomb"],
           "diesel": ["diesel", "gasoil", "c'est un diesel"],
           "hybride": ["hybride", "c'est une hybride", "hybride essence-électrique"],
           "electrique": ["électrique", "100 % électrique", "elle est électrique"]}
ACQUISITION = {"comptant": ["comptant", "payé cash", "je l'ai acheté comptant", "acheté sans crédit"],
               "credit": ["à crédit", "avec un prêt", "je l'ai acheté avec un crédit auto", "crédit sur 4 ans"],
               "LOA_LLD": ["en LOA", "en location longue durée", "en leasing", "LLD"]}
USAGE = {"prive": ["juste pour le privé, courses et week-ends", "usage privé seulement", "perso uniquement, pas pour le boulot"],
         "prive_trajet_travail": ["privé et pour aller au travail", "pour aller au boulot et le perso", "trajet travail plus les week-ends",
                                  "domicile-travail tous les jours"],
         "professionnel": ["pour le travail, je vois des clients", "usage professionnel", "je m'en sers pour mon métier, déplacements pro"],
         "loisirs": ["que pour les loisirs", "les balades le week-end, c'est tout", "uniquement loisirs, l'été surtout"],
         "vtc": ["pour faire du VTC", "transport de clients en VTC"]}
STATIONNEMENT = {"garage_ferme": ["dans mon garage", "garage fermé", "dans un box fermé", "au garage"],
                 "parking_prive": ["sur le parking de la résidence", "place de parking privée", "dans le parking de l'immeuble",
                                   "dans la cour, c'est privé"],
                 "voie_publique": ["dans la rue", "sur la voie publique", "devant chez moi, dans la rue", "sur le trottoir"]}
SECONDAIRE = {"aucun": ["non, que moi", "personne d'autre", "non, je suis le seul à la conduire", "juste moi"],
              "conjoint": ["ma femme aussi", "mon mari la prend parfois", "ma conjointe", "mon compagnon aussi"],
              "enfant_jeune_conducteur": ["mon fils, il vient d'avoir son permis", "ma fille, jeune conductrice",
                                          "mon fils de 19 ans de temps en temps"],
              "parent": ["mon père parfois", "ma mère de temps en temps", "mes parents l'empruntent"]}
FORMULE = {"tiers": ["au tiers", "le minimum, au tiers", "juste la responsabilité civile", "la formule la moins chère, au tiers"],
           "tiers_plus": ["tiers plus, avec le vol et le bris de glace", "une formule intermédiaire", "tiers étendu",
                          "entre les deux, tiers avec vol et incendie"],
           "tous_risques": ["tous risques", "la meilleure couverture", "tous risques, je veux être bien couvert", "une tous risques"]}
FRANCHISE = {"basse": ["une franchise basse", "la plus petite franchise possible", "franchise faible, quitte à payer un peu plus"],
             "standard": ["une franchise normale", "standard", "la franchise classique", "peu importe, standard"],
             "haute": ["une franchise haute si ça baisse le prix", "franchise élevée", "grosse franchise, pour payer moins chaque mois"]}
PAIEMENT = {"mensuel": ["tous les mois", "mensuellement", "au mois", "par mois"],
            "annuel": ["une fois par an", "annuellement", "en une fois", "à l'année"]}
RESILIATION = {"non": ["non, jamais", "non", "jamais"],
               "sinistres": ["oui, pour sinistres", "oui, après plusieurs accidents", "oui, résilié pour sinistralité"],
               "non_paiement": ["oui, pour non-paiement", "oui, j'avais des impayés", "oui, pour défaut de paiement"],
               "fausse_declaration": ["oui, pour fausse déclaration", "oui, une histoire de fausse déclaration"]}

POLI = ["Très bien.", "D'accord.", "Parfait.", "Merci.", "Noté.", "Entendu.", "Super.", ""]
REMPLISSAGE = ["", "", "", "Alors ", "Euh ", "Ben ", "Bah "]


def permis_normalise(t: str) -> str:
    if t.startswith("A2"):
        return "A2"
    if t.startswith("B (") or t == "B":
        return "B"
    return t


def bonus_dit(p) -> str:
    """Comment le client dit son bonus-malus, selon le système de son pays."""
    n = p.niveau_bonus_malus_national
    s = p.systeme_bonus_malus_national
    if s == "crm_france":
        c = p.coefficient_bonus_malus
        if c < 1 and chance(0.5):
            return un([f"j'ai {round((1 - c) * 100)} % de bonus", f"bonus de {round((1 - c) * 100)} %"])
        if c > 1 and chance(0.5):
            return un([f"j'ai un malus, coefficient {str(c).replace('.', ',')}", f"malus de {round((c - 1) * 100)} %"])
        return un([f"mon coefficient est de {c:.2f}".replace(".", ","), f"CRM {c:.2f}", f"{c:.2f} de bonus-malus"])
    if s == "sf_klasse":
        if n.startswith("M"):
            return un(["je suis en malus", "classe M, en malus"])
        return un([f"je suis en {n}", f"SF-Klasse {n.split()[-1]}", f"classe {n}"])
    if s == "classe_merito":
        return un([f"je suis en {n}", f"{n} de mérite", f"classe di merito {n.split()[-1]}"])
    if s == "degre_bonus_malus":
        return un([f"je suis au {n}", f"{n} sur l'échelle bonus-malus", f"degré {n.split()[-1]}"])
    if s == "stufe_autriche":
        return un([f"je suis en {n}", f"Bonus-Malus {n}", f"échelon {n.split()[-1]} ({n})"])
    if s == "trede":
        return un([f"je suis à la {n}", f"{n} sur l'échelle", f"bonus-malus {n}"])
    if s == "bonus_malus_roumanie":
        return un([f"je suis en classe {n}", f"{n}", f"classe bonus-malus {n}"])
    if s in ("bonus_norvege", "bonus_finlande"):
        return un([f"j'ai {n.replace('bonus ', '')} de bonus", f"{n}", f"un {n}"])
    if s == "no_claims_bonus":
        return un([f"j'ai {n}", f"{n.replace(' de no claims bonus', '')} de no claims bonus", f"{n} (NCB)"])
    return un([f"{n}", f"ça fait {n}", f"j'ai {n}"])


def sinistres_dit(r: int, nr: int) -> str:
    if r == 0 and nr == 0:
        return un(["aucun sinistre", "non, aucun", "rien du tout", "jamais eu d'accident"])
    morceaux = []
    if r:
        morceaux.append(un([f"{r} responsable" + ("s" if r > 1 else ""), f"{r} où j'étais en tort" if r == 1 else f"{r} où j'étais en tort",
                            f"{r} accident{'s' if r > 1 else ''} responsable{'s' if r > 1 else ''}"]))
    else:
        morceaux.append(un(["aucun responsable", "aucun de ma faute"]))
    if nr:
        morceaux.append(un([f"{nr} non responsable" + ("s" if nr > 1 else ""), f"{nr} où ce n'était pas ma faute"]))
    else:
        morceaux.append(un(["aucun non responsable", "et rien d'autre"]))
    return ", ".join(morceaux)


# --------------------------------------------------------------------------- étapes de la conversation
# chaque étape : (nom, questions possibles, fonction(profil) -> (réponse, {champ: valeur}))
def e_vehicule(p):
    moto = p.type_vehicule == "moto"
    nom = "ma moto" if moto and "scooter" not in p.categorie_vehicule else ("mon scooter" if moto else "ma voiture")
    champs = {"type_vehicule": p.type_vehicule, "marque_modele": p.marque_modele}
    t = un([f"{nom}, une {p.marque_modele}", f"c'est pour {nom}, une {p.marque_modele}", f"une {p.marque_modele}",
            f"{p.marque_modele}, c'est {nom}"])
    if not moto and "ma voiture" not in t and chance(0.5):
        t = f"une voiture, une {p.marque_modele}"
    if chance(0.3):
        t += f", {un(ENERGIE[p.energie])}"
        champs["energie"] = p.energie
    if chance(0.25):
        t += f", de {p.annee_mise_en_circulation}"
        champs["annee_mise_en_circulation"] = int(p.annee_mise_en_circulation)
    return t, champs


def e_energie(p):
    return un(ENERGIE[p.energie]).capitalize(), {"energie": p.energie}


def e_annee(p):
    a = int(p.annee_mise_en_circulation)
    t = un([f"de {a}", f"{a}", f"première mise en circulation en {a}", f"elle a {ans(ANNEE - a)}" if ANNEE - a > 0 else "elle est neuve"])
    return t, {"annee_mise_en_circulation": a}


def e_acquisition(p):
    return un(ACQUISITION[p.mode_acquisition]), {"mode_acquisition": p.mode_acquisition}


def e_age(p):
    return un([f"{p.age} ans", f"j'ai {p.age} ans", f"{p.age}"]), {"age": int(p.age)}


def e_situation(p):
    return un(SITUATION[p.situation_familiale]), {"situation_familiale": p.situation_familiale}


def e_profession(p):
    return un(PROFESSION[p.profession]), {"profession": p.profession}


def e_lieu(p):
    c, pays = p.commune, p.pays_residence
    t = un([f"à {c}", f"j'habite à {c}, {prep_pays(pays)}", f"{c}, {prep_pays(pays)}", f"{c} ({p.code_postal})",
            f"à {c} {prep_pays(pays)}", f"sur {c}"])
    return t, {"commune": c, "pays_residence": pays}


def e_permis(p):
    tp = permis_normalise(p.type_permis)
    n = int(p.annees_permis)
    annee = ANNEE - n
    nom = {"B": "le permis B", "B+formation": "le permis B avec la formation 125", "A": "le permis A", "A2": "le permis A2",
           "A1": "le permis A1", "AM": "le permis AM (BSR)"}.get(tp, f"le permis {tp}")
    if n == 0:
        t = un([f"{nom}, je viens de l'avoir", f"{nom}, obtenu cette année", f"{nom} depuis quelques mois"])
    else:
        t = un([f"{nom} depuis {annee}", f"{nom}, ça fait {ans(n)}", f"{nom}, obtenu en {annee}", f"{tp}, depuis {ans(n)}"])
    return t, {"type_permis": tp, "annees_permis": n}


def e_pays_permis(p):
    if p.pays_permis == p.pays_residence:
        t = un([f"{prep_pays(p.pays_permis)}", "ici", f"ici, {prep_pays(p.pays_permis)}", "dans le pays où j'habite"])
    else:
        t = un([f"{prep_pays(p.pays_permis)}", f"je l'ai passé {prep_pays(p.pays_permis)}",
                f"{prep_pays(p.pays_permis)}, il est " + ("reconnu ici" if p.permis_echange_ou_reconnu else "pas encore échangé")])
    return t, {"pays_permis": p.pays_permis}


def e_assure(p):
    if not p.deja_assure:
        return un(["non, c'est ma première assurance", "non jamais", "non, je n'ai jamais été assuré à mon nom",
                   "jamais comme conducteur principal"]), {"deja_assure": False, "annees_assure_total": 0}
    n = int(p.annees_assure_total)
    return un([f"oui, depuis {ans(n)}", f"oui, ça fait {ans(n)}", f"oui, {ans(n)} d'assurance"]), \
        {"deja_assure": True, "annees_assure_total": n}


def e_bonus(p):
    return bonus_dit(p), {"bonus_malus": p.niveau_bonus_malus_national}


def e_sinistres(p):
    r, nr = int(p.sinistres_responsables_36_mois), int(p.sinistres_non_responsables_36_mois)
    return sinistres_dit(r, nr), {"sinistres_responsables_36_mois": r, "sinistres_non_responsables_36_mois": nr}


def e_resiliation(p):
    return un(RESILIATION[p.resiliation_assureur]), {"resiliation_assureur": p.resiliation_assureur}


def e_usage(p):
    return un(USAGE[p.usage]), {"usage": p.usage}


def e_km(p):
    k = int(p.kilometrage_annuel)
    return un([f"environ {nombre(k)} km par an", f"{nombre(k)} km", f"à peu près {nombre(k)} kilomètres par an", f"{nombre(k)}"]), \
        {"kilometrage_annuel": k}


def e_stationnement(p):
    return un(STATIONNEMENT[p.stationnement_nuit]), {"stationnement_nuit": p.stationnement_nuit}


def e_secondaire(p):
    return un(SECONDAIRE[p.conducteur_secondaire]), {"conducteur_secondaire": p.conducteur_secondaire}


def e_formule(p):
    return un(FORMULE[p.formule_souhaitee]), {"formule_souhaitee": p.formule_souhaitee}


def e_franchise(p):
    return un(FRANCHISE[p.franchise_souhaitee]), {"franchise_souhaitee": p.franchise_souhaitee}


def e_budget(p):
    b, d = int(p.budget_mensuel_max_devise_locale), p.devise_locale
    mot = un(DEVISES[d])
    sym = mot in ("€", "£")
    m = f"{b}{mot}" if sym else f"{b} {mot}"
    return un([f"pas plus de {m} par mois", f"{m} max par mois", f"autour de {m} par mois", f"{m} par mois maximum"]), \
        {"budget_mensuel_max": b, "devise_budget": d}


def e_paiement(p):
    return un(PAIEMENT[p.paiement]), {"paiement": p.paiement}


ETAPES = [
    ("vehicule", ["Quel véhicule souhaitez-vous assurer ?", "C'est pour quel véhicule ?", "Vous voulez assurer quoi exactement ?",
                  "Quel est le véhicule à assurer ?"], e_vehicule),
    ("energie", ["Quelle est sa motorisation ?", "Il roule à quoi : essence, diesel, hybride, électrique ?", "Quel carburant ?"], e_energie),
    ("annee", ["De quelle année est le véhicule ?", "Quelle est sa date de première mise en circulation ?", "Il date de quand ?"], e_annee),
    ("acquisition", ["Comment l'avez-vous financé ?", "Acheté comptant, à crédit ou en location ?", "Vous l'avez payé comment ?"], e_acquisition),
    ("age", ["Quel âge avez-vous ?", "Vous avez quel âge ?", "Pouvez-vous me donner votre âge ?"], e_age),
    ("situation", ["Quelle est votre situation familiale ?", "Vous êtes marié, en couple, célibataire ?"], e_situation),
    ("profession", ["Quelle est votre profession ?", "Vous faites quoi dans la vie ?", "Quelle est votre activité professionnelle ?"], e_profession),
    ("lieu", ["Où habitez-vous ?", "Dans quelle ville résidez-vous ?", "Vous habitez où ?"], e_lieu),
    ("permis", ["Quel permis avez-vous, et depuis quand ?", "Depuis quand avez-vous votre permis ?", "Quel type de permis, obtenu quand ?"], e_permis),
    ("pays_permis", ["Dans quel pays avez-vous obtenu votre permis ?", "Votre permis a été passé où ?"], e_pays_permis),
    ("assure", ["Avez-vous déjà été assuré en tant que conducteur principal ?", "Vous avez déjà eu une assurance à votre nom ?"], e_assure),
    ("bonus", ["Quel est votre bonus-malus ?", "Qu'indique votre relevé d'information pour le bonus-malus ?", "Vous êtes à combien de bonus ?"], e_bonus),
    ("sinistres", ["Avez-vous eu des sinistres ces trois dernières années ?", "Des accidents ces 36 derniers mois ?",
                   "Combien de sinistres, responsables ou non, sur les 3 dernières années ?"], e_sinistres),
    ("resiliation", ["Avez-vous déjà été résilié par un assureur ?", "Un assureur a-t-il déjà mis fin à votre contrat ?"], e_resiliation),
    ("usage", ["Quel usage faites-vous du véhicule ?", "Vous l'utilisez pour quoi ?", "Usage privé, trajet travail ou professionnel ?"], e_usage),
    ("km", ["Combien de kilomètres faites-vous par an ?", "Vous roulez combien par an ?", "Quel kilométrage annuel ?"], e_km),
    ("stationnement", ["Où est-il garé la nuit ?", "Il dort où la nuit ?", "Où stationnez-vous le véhicule la nuit ?"], e_stationnement),
    ("secondaire", ["Y a-t-il un conducteur secondaire ?", "Quelqu'un d'autre le conduit ?", "D'autres conducteurs à déclarer ?"], e_secondaire),
    ("formule", ["Quelle formule souhaitez-vous ?", "Vous cherchez plutôt tiers ou tous risques ?", "Quel niveau de couverture voulez-vous ?"], e_formule),
    ("franchise", ["Quelle franchise préférez-vous ?", "Plutôt une franchise basse, standard ou haute ?"], e_franchise),
    ("budget", ["Quel est votre budget maximum par mois ?", "Vous voulez payer combien au maximum par mois ?", "Quel budget avez-vous en tête ?"], e_budget),
    ("paiement", ["Vous préférez payer au mois ou à l'année ?", "Paiement mensuel ou annuel ?"], e_paiement),
]
CHAMPS_ETAPE = {"vehicule": ["marque_modele"], "energie": ["energie"], "annee": ["annee_mise_en_circulation"],
                "acquisition": ["mode_acquisition"], "age": ["age"], "situation": ["situation_familiale"],
                "profession": ["profession"], "lieu": ["commune"], "permis": ["type_permis"], "pays_permis": ["pays_permis"],
                "assure": ["deja_assure"], "bonus": ["bonus_malus"], "sinistres": ["sinistres_responsables_36_mois"],
                "resiliation": ["resiliation_assureur"], "usage": ["usage"], "km": ["kilometrage_annuel"],
                "stationnement": ["stationnement_nuit"], "secondaire": ["conducteur_secondaire"], "formule": ["formule_souhaitee"],
                "franchise": ["franchise_souhaitee"], "budget": ["budget_mensuel_max"], "paiement": ["paiement"]}
PEUT_IGNORER = {"annee", "km", "franchise", "bonus", "acquisition"}  # le client peut ne pas savoir
CORRIGEABLE = {"age": "age", "km": "kilometrage_annuel", "annee": "annee_mise_en_circulation"}


def mauvaise_valeur(champ, v):
    if champ == "age":
        return v + un([-2, -1, 1, 2])
    if champ == "kilometrage_annuel":
        return max(1000, v + un([-3000, -2000, 2000, 5000]))
    return v + un([-1, 1])


def conversation(p, couper: bool):
    """Retourne (tours [(qui, texte)], profil attendu)."""
    profil = {k: None for k in NOMS}
    tours = []
    etapes = list(ETAPES)
    # l'ordre change un peu d'une conversation à l'autre (les blocs restent logiques)
    if chance(0.4):
        bloc_conducteur = etapes[4:8]
        R.shuffle(bloc_conducteur)
        etapes = etapes[:4] + bloc_conducteur + etapes[8:]
    if chance(0.3):
        etapes = etapes[4:8] + etapes[:4] + etapes[8:]
    # étapes sans objet
    if not p.deja_assure:
        etapes = [e for e in etapes if e[0] not in ("bonus", "sinistres", "resiliation")]
    faites = set()

    tours.append(("Conseiller", un(["Bonjour, je vais vous aider à trouver la bonne assurance. ",
                                    "Bonjour ! Je vais vous poser quelques questions pour votre devis. ",
                                    "Bonjour, bienvenue. Faisons votre devis ensemble. "]) +
                  un(["Pour commencer, qu'est-ce que vous souhaitez assurer ?", "Que puis-je faire pour vous ?",
                      "Parlez-moi un peu de votre besoin."])))
    # le client donne souvent plusieurs informations d'un coup
    if chance(0.45):
        candidats = [e for e in etapes if e[0] in ("vehicule", "age", "lieu", "profession", "formule", "permis", "usage")]
        choix = R.sample(candidats, k=min(len(candidats), R.randint(2, 4)))
        if not any(c[0] == "vehicule" for c in choix):
            choix.insert(0, next(e for e in etapes if e[0] == "vehicule"))
        morceaux = []
        for nom, _, f in choix:
            t, champs = f(p)
            morceaux.append(t)
            profil.update(champs)
            faites.add(nom)
        tours.append(("Client", un(["Bonjour, ", "Bonjour. ", "Salut, ", ""]) + un(["je cherche une assurance : ", "alors voilà : ", ""]) +
                      ", ".join(morceaux) + "."))
    else:
        tours.append(("Client", un(["Bonjour, je voudrais un devis.", "Bonjour, je cherche une assurance.", "Je veux assurer mon véhicule.",
                                    "Bonjour, c'est pour un devis s'il vous plaît."])))

    restantes = [e for e in etapes if e[0] not in faites]
    stop = len(restantes) if not couper else R.randint(1, max(1, len(restantes) - 1))
    for i, (nom, questions, f) in enumerate(restantes[:stop]):
        if all(profil[c] is not None for c in CHAMPS_ETAPE[nom]):
            continue  # déjà dit plus tôt dans la conversation
        if chance(0.04):
            continue  # le conseiller oublie cette question
        q = un(questions)
        poli = un(POLI)
        tours.append(("Conseiller", f"{poli} {q}".strip() if i else q))
        if nom in PEUT_IGNORER and chance(0.05):
            tours.append(("Client", un(["Je ne sais pas trop.", "Aucune idée, je dois vérifier.", "Je ne suis pas sûr, je regarderai."])))
            tours.append(("Conseiller", un(["Pas de souci, on pourra compléter plus tard.", "D'accord, on y reviendra."])))
            continue
        t, champs = f(p)
        cle = CORRIGEABLE.get(nom)
        if cle and chance(0.06):
            faux = mauvaise_valeur(cle, champs[cle])
            t_faux = t.replace(str(champs[cle]), str(faux), 1) if str(champs[cle]) in t else str(faux)
            tours.append(("Client", t_faux[:1].upper() + t_faux[1:] + "."))
            tours.append(("Conseiller", un(["C'est noté.", "D'accord."])))
            t = un(["Ah non pardon, je me suis trompé : ", "Attendez, en fait c'est ", "Correction : "]) + t
        elif chance(0.15):
            t = un(REMPLISSAGE) + t
        if chance(0.1):
            t = t.lower()
        tours.append(("Client", t[:1].upper() + t[1:] + ("" if t.endswith((".", "!", "?")) else ".")))
        profil.update(champs)
        # une question du client de temps en temps
        if chance(0.04):
            tours.append(("Client", un(["C'est obligatoire de répondre ?", "Ça change beaucoup le prix ?", "Pourquoi vous demandez ça ?"])))
            tours.append(("Conseiller", un(["Oui, c'est un critère que tous les assureurs demandent.",
                                            "Ça joue sur le tarif, c'est pour vous proposer un prix juste."])))
    if not couper:
        tours.append(("Conseiller", un(["Merci, j'ai tout ce qu'il me faut. Je vous prépare les offres.",
                                        "Parfait, je compare les offres pour vous.", "C'est complet, merci !"])))
    return tours, profil


def texte(tours):
    return "\n".join(f"{qui} : {t}" for qui, t in tours)


def exemple(p, couper):
    tours, profil = conversation(p, couper)
    conv = texte(tours)
    return {
        "id_profil": p.id_profil, "coupee": couper, "tours": tours, "profil_attendu": profil,
        "messages": [{"role": "system", "content": CONSIGNE},
                     {"role": "user", "content": message_utilisateur(conv)},
                     {"role": "assistant", "content": json.dumps(profil, ensure_ascii=False)}],
    }


def main() -> int:
    SORTIE.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROFILS)
    df = df[df.age >= 16]  # hors cyclomoteurs à 14-15 ans, rares dans un comparateur
    tirage = df.sample(N_TRAIN + N_VALID + N_TEST, random_state=GRAINE)
    jeux = {"train": tirage.iloc[:N_TRAIN], "valid": tirage.iloc[N_TRAIN:N_TRAIN + N_VALID], "test": tirage.iloc[N_TRAIN + N_VALID:]}
    for nom, part in jeux.items():
        exemples = [exemple(p, chance(0.2)) for p in part.itertuples()]
        with open(SORTIE / f"{nom}.jsonl", "w", encoding="utf-8") as f:
            for e in exemples:
                f.write(json.dumps({"messages": e["messages"]}, ensure_ascii=False) + "\n")
        if nom == "test":
            with open(SORTIE / "test_conversations.jsonl", "w", encoding="utf-8") as f:
                for e in exemples:
                    f.write(json.dumps({k: e[k] for k in ("id_profil", "coupee", "tours", "profil_attendu")}, ensure_ascii=False) + "\n")
        print(f"{nom} : {len(exemples)} conversations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
