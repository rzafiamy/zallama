#!/usr/bin/env python3
"""
embed_bench.py — Retrieval smoke test for registered embedding models.

For each embedding-modality model in the registry, embeds a handful of
query/passage pairs (English, French, and a cross-lingual case) plus three
distractor passages per query, and checks that cosine similarity ranks the
correct passage above every distractor. This won't catch subtle quality
differences (see MTEB for that) but it catches the things that actually bite
in practice: a model that fails to load, a broken GGUF conversion, or a
language the model just doesn't handle.

Hits the live Zallama daemon's /v1/embeddings — the daemon must be running
and will lazy-load each model on demand, evicting others in its
`services` group as needed. A model that's much bigger than what fits
alongside your other loaded models may 503 with a CUDA OOM; unload large
text models first (`zallama unload <model>`) to free VRAM.

Usage:
    python3 scripts/embed_bench.py                  # every registered embedding model
    python3 scripts/embed_bench.py granite qwen      # substring match on name
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from server.config import load_config  # noqa: E402

# (lang, query, correct passage, [distractor passages...])
CASES = [
    ("EN", "How do I reset my password?",
     "To reset your password, click 'Forgot password' on the login page and follow the emailed link.",
     ["The weather today is sunny with a light breeze.",
      "Our return policy allows refunds within 30 days of purchase.",
      "The French Revolution began in 1789."]),
    ("EN", "What are the health benefits of drinking green tea?",
     "Green tea is rich in antioxidants called catechins, which may reduce inflammation and support heart health.",
     ["The stock market fell sharply today amid inflation fears.",
      "To install the software, download the package and run the installer.",
      "The Eiffel Tower is located in Paris, France."]),
    ("FR", "Comment réinitialiser mon mot de passe ?",
     "Pour réinitialiser votre mot de passe, cliquez sur « mot de passe oublié » sur la page de connexion et suivez le lien reçu par e-mail.",
     ["La météo aujourd'hui est ensoleillée avec une légère brise.",
      "Notre politique de retour permet un remboursement dans les 30 jours suivant l'achat.",
      "La Révolution française a commencé en 1789."]),
    ("FR", "Quels sont les bienfaits du thé vert pour la santé ?",
     "Le thé vert est riche en antioxydants appelés catéchines, qui peuvent réduire l'inflammation et favoriser la santé cardiaque.",
     ["La bourse a fortement chuté aujourd'hui en raison des craintes d'inflation.",
      "Pour installer le logiciel, téléchargez le paquet et exécutez l'installeur.",
      "La tour Eiffel est située à Paris, en France."]),
    ("FR-EN cross", "What is the capital of France?",
     "La capitale de la France est Paris, une ville connue pour la tour Eiffel.",
     ["Le chat dort sur le canapé toute la journée.",
      "Les recettes de cuisine italienne utilisent souvent de la tomate et du basilic.",
      "Le prix du pétrole a augmenté ce mois-ci."]),
]


def zallama_host() -> str:
    cfg = load_config()["zallama"]
    host = cfg["host"]
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{cfg['port']}"


def embedding_models(base: str) -> list[str]:
    r = requests.get(f"{base}/api/models", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", []) if m.get("modality") == "embedding"]


def embed(base: str, model: str, text: str) -> list[float]:
    r = requests.post(f"{base}/v1/embeddings", json={"model": model, "input": text}, timeout=120)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def run(base: str, model: str) -> None:
    print(f"\n=== {model} ===")
    correct, total, margins = 0, 0, []
    for lang, query, passage, distractors in CASES:
        try:
            qv = embed(base, model, query)
            pv = embed(base, model, passage)
            dvs = [embed(base, model, d) for d in distractors]
        except requests.HTTPError as e:
            print(f"  [{lang}] ERROR: {e}")
            continue
        sim_correct = cosine(qv, pv)
        sim_best_distractor = max(cosine(qv, dv) for dv in dvs)
        margin = sim_correct - sim_best_distractor
        ok = margin > 0
        total += 1
        correct += int(ok)
        margins.append(margin)
        print(f"  [{lang}] correct={sim_correct:.4f} best_distractor={sim_best_distractor:.4f} "
              f"margin={margin:+.4f} {'OK' if ok else 'FAIL'}")
    avg_margin = sum(margins) / len(margins) if margins else 0.0
    print(f"  --> {correct}/{total} correct rankings, avg margin {avg_margin:+.4f}")


def main() -> None:
    base = zallama_host()
    filters = sys.argv[1:]
    try:
        models = embedding_models(base)
    except requests.RequestException as e:
        print(f"Cannot reach Zallama daemon at {base}: {e}", file=sys.stderr)
        sys.exit(1)
    if filters:
        models = [m for m in models if any(f.lower() in m.lower() for f in filters)]
    if not models:
        print("No matching embedding-modality models found in the registry.", file=sys.stderr)
        sys.exit(1)
    for model in models:
        run(base, model)


if __name__ == "__main__":
    main()
