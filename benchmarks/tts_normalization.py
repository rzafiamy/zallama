"""Does text normalization make a TTS engine read numbers right?

For each TTS model given, synthesize every test sentence through zallama's
/v1/audio/speech, transcribe the audio with an ASR model, and score the
transcript against the original text. Both sides go through tn-server
(strict rules) first, so "9h30" written by the ASR and "neuf heures trente"
compare as the same words.

Usage:
  python3 benchmarks/tts_normalization.py --url http://localhost:6767 \
      --asr parakeet-tdt-v3-cpu --tn bin/tn-server \
      fr:pocket-tts-fr-raw fr:pocket-tts-fr en:kokoro:82m ...

A model is given as <lang>:<registry name>[@voice]; it runs the sentences of
that language (fr or en), each --repeats times (engines sample: one take is
noisy). Prints mean WER per model and per sentence.
"""
import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
import urllib.request

SENTENCES = {
    "fr": [
        "La réunion commence à 9h30 et le budget atteint 1 250 000 €.",
        "Le 1er mai, il fera entre 18 et 24 °C, soit 3,5 % de plus.",
        "Appelez le 06 12 34 56 78 avant le 21/10/2026, Mme Dupont.",
        "Au XVIIe siècle, Louis XIV régnait déjà depuis 1643.",
        "Le colis pèse 2,5 kg et coûte 12,99 €.",
        "Rendez-vous au 3e étage, salle n° 204, à 14h15.",
        "**Important** : la vitesse est limitée à 80 km/h depuis 2018.",
        "Le Dr Martin reçoit du lundi au vendredi, de 8h à 18h30.",
    ],
    "en": [
        "Meet at 9:30 am, it costs $5.50 and rose 12% in 1984.",
        "Dr. Smith lives at 221B Baker St., call (555) 123-4567 on Oct. 21, 2026.",
        "The 3rd quarter revenue was $2.4 million, up 7.5% from Q2.",
        "It weighs 3.2 kg and measures 45 cm.",
        "World War II ended in 1945; Henry VIII died in 1547.",
        "The flight leaves at 6:05 pm from gate 12B.",
        "**Note:** the temperature dropped to -5°C overnight.",
        "Call 1-800-555-0199 before Jan. 1st.",
    ],
}


def post_json(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), dict(r.headers)


def transcribe(url, asr, wav):
    boundary = "----tnbench"
    data = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{asr}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url + "/v1/audio/transcriptions", data, {"content-type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r).get("text", "")


def canonical_digits(text):
    """Join digit groups: the ASR formats numbers its own way ("1-8005550-199"
    for "1-800-555-0199", "9 .30" for "9:30"), which is not a reading error."""
    prev = None
    while prev != text:
        prev, text = text, re.sub(r"(\d)(?:\s?[-.:]\s?|\s?,(?=\d)|\)\s?|\s\(?|\()(?=\d)", r"\1", text)
    return text


def tn(binary, lang, text):
    text = canonical_digits(text)
    return subprocess.run([binary, "normalize", "--lang", lang, text], capture_output=True, text=True).stdout


def words(s):
    s = unicodedata.normalize("NFC", s.lower()).replace("’", "'").replace("-", " ")
    s = re.sub(r"\b([ap])\.?\s?m\b\.?", r"\1m", s)
    return re.sub(r"[^\w' ]+", " ", s).split()


def wer(hyp, ref):
    h, r = words(hyp), words(ref)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6767")
    ap.add_argument("--asr-url", default=None, help="ASR zallama URL (default: --url)")
    ap.add_argument("--asr", default="parakeet-tdt-v3-cpu")
    ap.add_argument("--tn", default="bin/tn-server")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--save", help="write every transcript to this JSON file")
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    asr_url = a.asr_url or a.url
    saved = []
    for spec in a.models:
        lang, model = spec.split(":", 1)
        model, _, voice = model.partition("@")
        total, rows, t_all = 0.0, [], 0.0
        for text in SENTENCES[lang]:
            ws, heard_all = [], []
            for _ in range(a.repeats):
                body = {"model": model, "input": text}
                if voice:
                    body["voice"] = voice
                t0 = time.time()
                wav, headers = post_json(a.url + "/v1/audio/speech", body)
                t_all += time.time() - t0
                heard = transcribe(asr_url, a.asr, wav)
                ws.append(wer(tn(a.tn, lang, heard), tn(a.tn, lang, text)))
                heard_all.append(heard)
                saved.append({"model": spec, "text": text, "heard": heard, "wer": ws[-1],
                              "normalized": headers.get("x-zallama-normalized", "-")})
            w = sum(ws) / len(ws)
            total += w
            rows.append((w, headers.get("x-zallama-normalized", "-"), heard_all[ws.index(max(ws))]))
        n = len(SENTENCES[lang])
        t_all /= a.repeats
        print(f"== {model} ({lang}): WER {100 * total / n:.1f}%  {1000 * t_all / n:.0f} ms/sentence")
        for w, norm, heard in rows:
            print(f"  {100 * w:5.1f}% [{norm}] {heard}")
        sys.stdout.flush()
    if a.save:
        with open(a.save, "w") as f:
            json.dump(saved, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
