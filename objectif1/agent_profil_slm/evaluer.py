"""
Évalue un modèle sur le jeu de test : 500 conversations jamais vues à l'entraînement, avec le profil attendu.

Tous les modèles reçoivent la même consigne (schema.py) et la même conversation, ce qui permet de comparer :
  - le modèle de base sans entraînement (ex. qwen2.5:1.5b dans Ollama),
  - le modèle fine-tuné (MLX, Hugging Face ou Ollama),
  - le modèle d'Ismail, ou n'importe quel modèle Ollama.

Exemples (depuis la racine du dépôt) :
  python3 objectif1/agent_profil_slm/evaluer.py --backend ollama --modele qwen2.5:1.5b --nom qwen1.5b-base
  python3 objectif1/agent_profil_slm/evaluer.py --backend mlx --modele objectif1/agent_profil_slm/modeles/qwen2.5-1.5b-instruct/fusionne --nom qwen1.5b-finetune
  python3 objectif1/agent_profil_slm/evaluer.py --backend hf --modele .../fusionne --nom qwen1.5b-finetune     (GPU)
  python3 objectif1/agent_profil_slm/evaluer.py --comparer       (tableau de tous les résultats déjà calculés)

Mesures :
  json_valide          part des réponses qui sont un JSON lisible
  exactitude           part des champs donnés dans la conversation que le modèle a bien remplis
  oublis               part des champs donnés que le modèle a laissés vides
  inventions           part des champs jamais donnés que le modèle a remplis quand même (hallucination)
  profils_parfaits     part des conversations où tout le profil est juste
  latence              secondes par conversation (moyenne et 95e centile), et jetons générés par seconde
Sorties : objectif1/agent_profil_slm/resultats/<nom>/ (resume.json, par_champ.csv, reponses.jsonl)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from schema import CHAMPS, CONSIGNE, NOMS, message_utilisateur  # noqa: E402

DOSSIER = Path(__file__).parent
TEST = DOSSIER / "donnees" / "test_conversations.jsonl"
RESULTATS = DOSSIER / "resultats"


# --------------------------------------------------------------------------- appel des modèles
def charger_backend(backend: str, modele: str, url: str):
    """Retourne une fonction messages -> (texte, nb_jetons)."""
    if backend == "ollama":
        import urllib.request

        def appeler(messages):
            corps = json.dumps({"model": modele, "messages": messages, "stream": False,
                                "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 512}}).encode()
            req = urllib.request.Request(f"{url}/api/chat", data=corps, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                rep = json.loads(r.read())
            return rep["message"]["content"], rep.get("eval_count", 0)
        return appeler

    if backend == "mlx":
        from mlx_lm import generate, load
        m, tok = load(modele)

        def appeler(messages):
            invite = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            texte = generate(m, tok, prompt=invite, max_tokens=512)
            return texte, len(tok.encode(texte))
        return appeler

    if backend == "hf":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(modele)
        m = AutoModelForCausalLM.from_pretrained(modele, torch_dtype="auto", device_map="auto")

        def appeler(messages):
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(m.device)
            with torch.no_grad():
                out = m.generate(ids, max_new_tokens=512, do_sample=False)
            gen = out[0, ids.shape[1]:]
            return tok.decode(gen, skip_special_tokens=True), len(gen)
        return appeler

    raise SystemExit(f"backend inconnu : {backend}")


# --------------------------------------------------------------------------- comparaison des valeurs
def lire_json(texte: str):
    texte = re.sub(r"^```(?:json)?|```$", "", texte.strip(), flags=re.M).strip()
    debut, fin = texte.find("{"), texte.rfind("}")
    if debut < 0 or fin < 0:
        return None
    try:
        d = json.loads(texte[debut:fin + 1])
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def simple(v) -> str:
    s = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s.replace(",", ".")).strip(" .")


def vide(v) -> bool:
    return v is None or (isinstance(v, str) and simple(v) in ("", "null", "none", "inconnu", "n/a"))


def egal(champ: str, attendu, predit) -> bool:
    if vide(predit):
        return False
    t = CHAMPS[champ][0]
    try:
        if t == "entier":
            return int(float(str(predit).replace(" ", "").replace(",", "."))) == int(attendu)
        if t == "booléen":
            return (simple(predit) in ("true", "oui", "1", "vrai")) == bool(attendu)
    except ValueError:
        return False
    return simple(predit) == simple(attendu)


# --------------------------------------------------------------------------- évaluation
def evaluer(args):
    exemples = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    if args.n:
        exemples = exemples[: args.n]
    appeler = charger_backend(args.backend, args.modele, args.url)
    dossier = RESULTATS / args.nom
    dossier.mkdir(parents=True, exist_ok=True)

    par_champ = {c: {"donnes": 0, "justes": 0, "oublis": 0, "non_donnes": 0, "inventions": 0} for c in NOMS}
    latences, vitesses, valides, parfaits = [], [], 0, 0
    with open(dossier / "reponses.jsonl", "w", encoding="utf-8") as sortie:
        for i, e in enumerate(exemples, 1):
            conv = "\n".join(f"{qui} : {t}" for qui, t in e["tours"])
            messages = [{"role": "system", "content": CONSIGNE}, {"role": "user", "content": message_utilisateur(conv)}]
            t0 = time.perf_counter()
            try:
                texte, n_jetons = appeler(messages)
            except Exception as err:  # un appel raté compte comme une réponse illisible
                texte, n_jetons = f"ERREUR : {err}", 0
            duree = time.perf_counter() - t0
            latences.append(duree)
            if n_jetons:
                vitesses.append(n_jetons / duree)
            pred = lire_json(texte)
            valides += pred is not None
            pred = pred or {}
            tout_juste = pred != {}
            for c in NOMS:
                att, p = e["profil_attendu"].get(c), pred.get(c)
                s = par_champ[c]
                if att is None:
                    s["non_donnes"] += 1
                    if not vide(p):
                        s["inventions"] += 1
                        tout_juste = False
                else:
                    s["donnes"] += 1
                    if egal(c, att, p):
                        s["justes"] += 1
                    else:
                        tout_juste = False
                        if vide(p):
                            s["oublis"] += 1
            parfaits += tout_juste
            sortie.write(json.dumps({"id_profil": e["id_profil"], "reponse": texte, "attendu": e["profil_attendu"],
                                     "duree_s": round(duree, 3)}, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(exemples):
                print(f"  {i}/{len(exemples)}  json valide {valides / i:.0%}  parfaits {parfaits / i:.0%}", flush=True)

    n = len(exemples)
    donnes = sum(s["donnes"] for s in par_champ.values())
    non_donnes = sum(s["non_donnes"] for s in par_champ.values())
    resume = {
        "nom": args.nom, "backend": args.backend, "modele": args.modele, "conversations": n,
        "json_valide": round(valides / n, 4),
        "exactitude": round(sum(s["justes"] for s in par_champ.values()) / donnes, 4),
        "oublis": round(sum(s["oublis"] for s in par_champ.values()) / donnes, 4),
        "inventions": round(sum(s["inventions"] for s in par_champ.values()) / max(1, non_donnes), 4),
        "profils_parfaits": round(parfaits / n, 4),
        "latence_moyenne_s": round(statistics.mean(latences), 3),
        "latence_p95_s": round(sorted(latences)[int(0.95 * (n - 1))], 3),
        "jetons_par_s": round(statistics.mean(vitesses), 1) if vitesses else None,
    }
    (dossier / "resume.json").write_text(json.dumps(resume, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(dossier / "par_champ.csv", "w", encoding="utf-8") as f:
        f.write("champ;exactitude;oublis;inventions;donnes;non_donnes\n")
        for c, s in par_champ.items():
            f.write(f"{c};{s['justes'] / max(1, s['donnes']):.3f};{s['oublis'] / max(1, s['donnes']):.3f};"
                    f"{s['inventions'] / max(1, s['non_donnes']):.3f};{s['donnes']};{s['non_donnes']}\n")
    print(json.dumps(resume, ensure_ascii=False, indent=2))


def comparer():
    lignes = []
    for f in sorted(RESULTATS.glob("*/resume.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        lignes.append(f"| {r['nom']} | {r['json_valide']:.0%} | {r['exactitude']:.1%} | {r['oublis']:.1%} | {r['inventions']:.1%} | "
                      f"{r['profils_parfaits']:.0%} | {r['latence_moyenne_s']:.2f} s | {r['jetons_par_s'] or '—'} |")
    tableau = ["| Modèle | JSON valide | Exactitude | Oublis | Inventions | Profils parfaits | Latence | Jetons/s |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"] + lignes
    (RESULTATS / "comparaison.md").write_text("\n".join(tableau) + "\n", encoding="utf-8")
    print("\n".join(tableau))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["ollama", "mlx", "hf"])
    ap.add_argument("--modele")
    ap.add_argument("--nom", help="nom du dossier de résultats")
    ap.add_argument("--n", type=int, default=0, help="évaluer seulement les n premières conversations (essai rapide)")
    ap.add_argument("--url", default="http://localhost:11434", help="adresse d'Ollama")
    ap.add_argument("--comparer", action="store_true")
    a = ap.parse_args()
    if a.comparer:
        comparer()
    else:
        if not (a.backend and a.modele and a.nom):
            ap.error("--backend, --modele et --nom sont nécessaires")
        evaluer(a)
