import hashlib
import html
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_classifier.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = 448

PAGES = ["Accueil", "Importer une image", "Scan caméra",
         "Tableau de bord", "Historique & insights"]
ICONS = [":material/home:", ":material/upload_file:", ":material/photo_camera:",
         ":material/agriculture:", ":material/insights:"]

st.set_page_config(page_title="Plant Health Guard", page_icon=":material/eco:", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;800&family=Inter:wght@400;500;600&display=swap');
:root{--g7:#15803d;--g6:#16a34a;--g1:#dcfce7;--g0:#f0fdf4;--ink:#0f172a;--sl:#64748b;--ln:#e5e7eb;--red:#dc2626;--amb:#d97706}
html,body,.stApp{font-family:'Inter',sans-serif;color:var(--ink);background:#f8fafc}
h1,h2,h3,.h{font-family:'Plus Jakarta Sans',sans-serif!important;letter-spacing:-.02em}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.block-container{padding-top:1.2rem;max-width:1300px}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--ln)}
.brand{display:flex;gap:.6rem;align-items:center;font-family:'Plus Jakarta Sans';font-weight:800;font-size:1.1rem;margin-bottom:1rem}
.logo{width:32px;height:32px;border-radius:9px;background:var(--g6);display:inline-flex;align-items:center;justify-content:center}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:.5rem .8rem;border-radius:9px;width:100%}
[data-testid="stSidebar"] [role="radiogroup"] label>div:first-child{display:none}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:var(--g1)}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{color:var(--g7);font-weight:700}
[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid var(--ln)!important;border-radius:16px!important;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.title{font-family:'Plus Jakarta Sans';font-weight:800;font-size:1.35rem;margin:.2rem 0 .6rem}
.card{background:#fff;border:1px solid var(--ln);border-radius:14px;padding:.9rem 1rem;height:190px;overflow:hidden}
.ct{font-weight:600;font-size:.85rem;margin-bottom:.5rem}
.kv{font-family:'Plus Jakarta Sans';font-weight:800;font-size:1.8rem;color:var(--g7)}
.ks{font-size:.8rem;color:var(--sl)}
.gauge{position:relative;width:200px;height:100px;margin:auto;overflow:hidden}
.gf{width:200px;height:200px;border-radius:50%}
.gi{position:absolute;left:30px;top:30px;width:140px;height:140px;border-radius:50%;background:#fff}
.gv{position:absolute;left:0;right:0;bottom:2px;text-align:center;font-family:'Plus Jakarta Sans';font-weight:800;font-size:1.5rem}
.vb{display:flex;align-items:flex-end;gap:14px;height:90px}
.vb div.c{flex:1;text-align:center;font-size:.72rem;color:var(--sl)}
.vb .b{border-radius:5px 5px 0 0;margin-bottom:4px}
.step{display:flex;gap:.8rem;margin:.6rem 0}
.n{flex:0 0 28px;height:28px;border-radius:50%;background:var(--g6);color:#fff;font-weight:700;display:inline-flex;align-items:center;justify-content:center;font-size:.85rem}
.sb{flex:1;border:1px solid var(--ln);border-radius:12px;padding:.7rem .9rem;background:#fff}
.sb.first{border-color:var(--g6);background:var(--g0)}
.sb b{font-size:.95rem}.sb span{display:block;color:var(--sl);font-size:.85rem;margin-top:.15rem}
.badge{padding:.15rem .6rem;border-radius:99px;font-size:.78rem;font-weight:700}
.ok{background:var(--g1);color:var(--g7)}.sick{background:#fee2e2;color:#991b1b}.warn{background:#fef3c7;color:#92400e}
.risk{height:10px;border-radius:99px;background:linear-gradient(90deg,#16a34a,#facc15,#dc2626);position:relative;margin:.9rem 0 .4rem}
.risk i{position:absolute;top:-4px;width:6px;height:18px;border-radius:3px;background:var(--ink)}
[data-baseweb="tab"]{font-weight:600}
[aria-selected="true"][data-baseweb="tab"]{color:var(--g7)}
[data-baseweb="tab-highlight"]{background:var(--g6)!important}
[data-testid="stFileUploaderDropzone"]{background:var(--g0);border:2px dashed var(--g6);border-radius:14px}
.stButton>button[kind="primary"]{background:var(--g6);border:none;color:#fff;font-weight:700;border-radius:10px;padding:.6rem}
.stButton>button[kind="primary"]:hover{background:var(--g7)}

.block-container{padding-top:3.5rem!important}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:#f8fafc!important;color:var(--ink)}
header[data-testid="stHeader"]{background:#f8fafc!important}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div{background:#fff!important}
[data-testid="stSidebar"] *,.stMarkdown,.stMarkdown p,label,[data-testid="stWidgetLabel"] *,[data-testid="stCaptionContainer"] *{color:var(--ink)}
.stButton>button,.stDownloadButton>button{background:#fff!important;color:var(--g7)!important;border:1px solid var(--g6)!important;border-radius:10px;font-weight:600}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--g0)!important}
.stButton>button[kind="primary"]{background:var(--g6)!important;color:#fff!important;border:none!important}
.stButton>button[kind="primary"] *{color:#fff!important}
[data-testid="stSidebar"] .stButton>button{justify-content:flex-start;background:transparent!important;border:none!important;color:var(--ink)!important;font-weight:500;box-shadow:none}
[data-testid="stSidebar"] .stButton>button *{color:inherit!important}
[data-testid="stSidebar"] .stButton>button:hover{background:var(--g0)!important}
[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:var(--g1)!important;color:var(--g7)!important;font-weight:700}
[data-baseweb="select"]>div{background:#fff!important;border:1px solid var(--ln)!important}
[data-baseweb="select"] *{color:var(--ink)!important}
[data-testid="stFileUploaderDropzone"] *{color:var(--ink)}
[data-testid="stFileUploaderDropzone"] button{background:#fff!important;border:1px solid var(--ln)!important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


class DiseaseClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(x)


@st.cache_resource
def load_models():
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet = resnet.to(DEVICE).eval()
    try:
        ck = torch.load(MODEL_PATH, map_location=DEVICE)
    except Exception:
        ck = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    clf = DiseaseClassifier(ck["input_size"], ck["num_classes"])
    clf.load_state_dict(ck["model_state_dict"])
    clf = clf.to(DEVICE).eval()
    for p in list(resnet.parameters()) + list(clf.parameters()):
        p.requires_grad_(False)
    return resnet, clf, ck["class_names"]


transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def parse_class_name(raw):
    if "___" in raw:
        plant, cond = raw.split("___", 1)
    elif "_" in raw:
        plant, cond = raw.split("_", 1)
    else:
        plant, cond = "Plante", raw
    plant, cond = plant.replace("_", " ").strip(), cond.replace("_", " ").strip()
    return plant, cond, ("healthy" in cond.lower() or "sain" in cond.lower())


ADVICE = [
    ("healthy", "Aucun signe de maladie. Continue la surveillance du feuillage."),
    ("blight", "Retire les feuilles atteintes, arrose au pied et aère les plants. Un fongicide adapté peut être utile."),
    ("rust", "Supprime les feuilles touchées et limite l'humidité sur le feuillage."),
    ("mildew", "Améliore la circulation d'air et retire les parties infectées."),
    ("mold", "Réduis l'humidité ambiante et retire les feuilles atteintes."),
    ("scab", "Détruis les feuilles tombées et taille pour aérer."),
    ("rot", "Retire les parties pourries, évite l'eau stagnante, désinfecte les outils."),
    ("spot", "Retire les feuilles tachées et évite les éclaboussures."),
    ("mite", "Inspecte le dessous des feuilles et envisage un acaricide."),
    ("virus", "Pas de traitement curatif : isole la plante et contrôle les insectes vecteurs."),
    ("mosaic", "Pas de traitement curatif : isole la plante et désinfecte les outils."),
    ("curl", "Souvent viral (mouche blanche) : isole la plante et contrôle les insectes."),
]


def get_advice(cond):
    for k, t in ADVICE:
        if k in cond.lower():
            return t
    return "Demande l'avis d'un agronome avant de traiter."


def get_plan(cond, healthy):
    if healthy:
        return {"Immédiat": [("Aucune intervention", ADVICE[0][1])],
                "Court terme": [("Suivi régulier", "Refais un scan toutes les 1 à 2 semaines.")],
                "Long terme": [("Prévention", "Espace les plants, arrose au pied, fais tourner les cultures.")]}
    return {"Immédiat": [("Traitement prioritaire", get_advice(cond)),
                         ("Isolement & taille", "Écarte les feuilles très atteintes, désinfecte les outils."),
                         ("Surveillance capteurs", "Contrôle humidité et température autour de la plante.")],
            "Court terme": [("Nouveau scan", "Refais une analyse dans 5 à 7 jours."),
                            ("Plantes voisines", "Inspecte les plantes proches.")],
            "Long terme": [("Gestion intégrée", "Rotation des cultures, variétés résistantes, bonne aération."),
                           ("Communauté", "Compare avec d'autres cas pour affiner ta stratégie.")]}


def compute_gradcam(x, idx, resnet, clf):
    st_ = {}

    def hook(_m, _i, out):
        st_["a"] = out
        out.register_hook(lambda g: st_.__setitem__("g", g))

    h = resnet.layer4.register_forward_hook(hook)
    try:
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            clf(resnet(x))[0, idx].backward()
    finally:
        h.remove()
    w = st_["g"].detach().mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * st_["a"].detach()).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).cpu().numpy()


@st.cache_data(show_spinner=False, max_entries=64)
def analyze(image_bytes, top_k=3):
    resnet, clf, names = load_models()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    mask = torch.tensor([parse_class_name(c)[2] for c in names], device=DEVICE)
    with torch.no_grad():
        p = torch.softmax(clf(resnet(x)), 1)
        p = (p + torch.softmax(clf(resnet(torch.flip(x, dims=[3]))), 1)) / 2
        k = min(top_k, len(names))
        tp, ti = torch.topk(p, k=k, dim=1)
        healthy = p[0][mask].sum().item() * 100
    preds = [(names[ti[0][i].item()], tp[0][i].item() * 100) for i in range(k)]
    cam = compute_gradcam(x, ti[0][0].item(), resnet, clf)
    hsv = np.asarray(img.resize((224, 224)).convert("HSV"), dtype=np.float32)
    leaf = float(((hsv[..., 0] >= 20) & (hsv[..., 0] <= 71) & (hsv[..., 1] > 77) & (hsv[..., 2] > 40)).mean())
    return {"preds": preds, "healthy": healthy, "cam": cam, "leaf": leaf, "image": img.resize((SIZE, SIZE))}


def peaks(cam, n=3, r=28):
    c, pts = cam.copy(), []
    for _ in range(n):
        y, x = np.unravel_index(c.argmax(), c.shape)
        if c[y, x] < 0.35:
            break
        pts.append((x / 224, y / 224))
        c[max(0, y - r):y + r, max(0, x - r):x + r] = 0
    return pts


def font(s):
    for f in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f, s)
        except Exception:
            pass
    return ImageFont.load_default()


def annotate(img, pts, lines, color):
    img = img.copy()
    d = ImageDraw.Draw(img)
    for i, (px, py) in enumerate(pts):
        cx, cy, s, L = int(px * SIZE), int(py * SIZE), 22, 9
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            ex, ey = cx + sx * s, cy + sy * s
            d.line([ex, ey, ex - sx * L, ey], fill=color, width=3)
            d.line([ex, ey, ex, ey - sy * L], fill=color, width=3)
        d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=color)
    if pts:
        f = font(13)
        w = max(d.textlength(t, font=f) for t in lines) + 16
        h = len(lines) * 18 + 10
        tx = min(max(6, int(pts[0][0] * SIZE) - w / 2), SIZE - w - 6)
        ty = max(6, int(pts[0][1] * SIZE) - 40 - h)
        d.rounded_rectangle([tx, ty, tx + w, ty + h], 8, fill=(30, 30, 30))
        for i, t in enumerate(lines):
            d.text((tx + 8, ty + 6 + i * 18), t, fill="white", font=f)
    return img


def heatmap(img, cam, op=0.6):
    base = np.asarray(img, dtype=np.float32)
    c = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255
    heat = np.stack([np.clip(1.5 - abs(4 * c - 3), 0, 1), np.clip(1.5 - abs(4 * c - 2), 0, 1),
                     np.clip(1.5 - abs(4 * c - 1), 0, 1)], -1) * 255
    w = (op * c)[..., None]
    return Image.fromarray((base * (1 - w) + heat * w).astype(np.uint8))


# ---------- composants HTML ----------
def card(title, body):
    return f'<div class="card"><div class="ct">{html.escape(title)}</div>{body}</div>'


def gauge(pct, color):
    a = pct * 1.8
    return ('<div class="gauge"><div class="gf" style="background:conic-gradient(from 270deg,'
            f'{color} 0deg {a:.1f}deg,#e5e7eb {a:.1f}deg 180deg,transparent 180deg 360deg)"></div>'
            f'<div class="gi"></div><div class="gv" style="color:{color}">{pct:.1f}%</div></div>')


def vigor(this, avg):
    bars = [("Cette feuille", this, "#16a34a"), ("Moyenne session", avg, "#86efac"), ("Référence", 100, "#d1d5db")]
    inner = "".join(f'<div class="c"><div class="b" style="height:{max(v, 3) * 0.8:.0f}px;background:{c}"></div>{n}</div>'
                    for n, v, c in bars)
    return f'<div class="vb">{inner}</div><div class="ks">Indice de vigueur vs plantes saines</div>'


def risk_card(score):
    if score is None:
        return '<span class="badge warn">Indéterminé</span><div class="ks" style="margin-top:.8rem">Image non fiable : risque non calculé.</div>'
    lvl = "Faible" if score < 33 else "Modéré" if score < 66 else "Élevé"
    cls = "ok" if score < 33 else "warn" if score < 66 else "sick"
    return (f'<span class="badge {cls}">{lvl}</span>'
            f'<div class="risk"><i style="left:calc({score:.0f}% - 3px)"></i></div>'
            '<div class="ks">Basé sur la probabilité de maladie et l\'étendue de la zone touchée.</div>')


def steps(items):
    out = ""
    for i, (t, d) in enumerate(items):
        out += (f'<div class="step"><span class="n">{i + 1}</span>'
                f'<div class="sb {"first" if i == 0 else ""}"><b>{html.escape(t)}</b>'
                f'<span>{html.escape(d)}</span></div></div>')
    return out


def go(p):
    st.session_state.page = p


# ---------- state ----------
st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("history", [])
st.session_state.setdefault("seen", set())

try:
    with st.spinner("Chargement du modèle..."):
        _, _, class_names = load_models()
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}\n\nChemin attendu : {MODEL_PATH}")
    st.stop()

threshold = 60  # seuil de confiance (%)

with st.sidebar:
    st.markdown('<div class="brand"><span class="logo"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.5 19 2c1 2 2 4.2 2 8 0 5.5-4.8 10-10 10Z"/><path d="M2 21c0-3 1.9-5.5 6-7"/></svg></span>Plant Health Guard</div>', unsafe_allow_html=True)
    for _p, _ic in zip(PAGES, ICONS):
        st.button(_p, icon=_ic, key=f"nav_{_p}", use_container_width=True,
                  type="primary" if st.session_state.page == _p else "secondary",
                  on_click=go, args=(_p,))


# ---------- diagnostic ----------
def render_result(name, content):
    d = analyze(content)
    best, prob = d["preds"][0]
    plant, cond, healthy = parse_class_name(best)
    cam = d["cam"]
    mask = cam >= 0.5
    area = float(mask.mean() * 100)
    sick = max(0.0, 100 - d["healthy"])
    noleaf = d["leaf"] < 0.08  # très peu de pixels vert/jaune/brun : probablement pas une feuille
    uncertain = prob < threshold or noleaf
    risk = None if uncertain else 0 if healthy else min(100, 0.6 * sick + 0.4 * min(area * 2, 100))
    color = "#16a34a" if prob >= max(threshold, 85) else "#d97706" if prob >= threshold else "#dc2626"

    key = hashlib.md5(content).hexdigest()
    if key not in st.session_state.seen and not uncertain:
        st.session_state.seen.add(key)
        st.session_state.history.append({
            "Heure": datetime.now().strftime("%H:%M:%S"), "Fichier": name, "Plante": plant,
            "État": cond, "Sain": "Oui" if healthy else "Non", "Confiance (%)": round(prob, 1),
            "Indice de santé (%)": round(d["healthy"], 1), "Zone (%)": round(area, 1),
            "Risque (%)": round(risk)})

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<div class="title">2. Résultats du diagnostic</div>', unsafe_allow_html=True)
        lines = ([f"{plant} · {cond}", f"Zone d'attention : {area:.1f}%", f"Probabilité de maladie : {sick:.0f}%"]
                 if not uncertain else
                 ["Aucune feuille détectée" if noleaf else "Résultat incertain", f"Hypothèse : {plant} · {cond}", f"Confiance : {prob:.0f}%"])
        annotated = annotate(d["image"], peaks(cam), lines, "#d97706" if uncertain else "#16a34a" if healthy else "#dc2626")
        t1, t2, t3 = st.tabs(["Annoté", "Carte de chaleur", "Original"])
        t1.image(annotated, use_container_width=True)
        t2.image(heatmap(d["image"], cam), use_container_width=True)
        t3.image(d["image"], use_container_width=True)
    with right:
        st.markdown('<div class="title">Recommandations & actions</div>', unsafe_allow_html=True)
        badge = ('<span class="badge warn">Incertain</span>' if uncertain else
                 '<span class="badge ok">Saine</span>' if healthy else '<span class="badge sick">Malade</span>')
        st.markdown(f"{badge} &nbsp; **{html.escape(plant)}** — {html.escape(cond)}", unsafe_allow_html=True)
        if uncertain:
            st.warning(("Aucune feuille détectée dans cette image (trop peu de zones végétales). " if noleaf else
                        "Confiance trop faible : l'image n'est probablement pas une feuille connue du modèle. ")
                       + "Reprends la photo : une seule feuille, nette, bien éclairée, sur fond simple, sans main ni objet.")
        else:
            plan = get_plan(cond, healthy)
            for tab, h in zip(st.tabs(list(plan)), plan):
                tab.markdown(steps(plan[h]), unsafe_allow_html=True)
            st.caption("Conseils généraux, à confirmer avec un spécialiste.")

    c1, c2, c3 = st.columns(3)
    c1.markdown(card("Score de diagnostic IA", gauge(prob, color)), unsafe_allow_html=True)
    avg = (pd.DataFrame(st.session_state.history)["Indice de santé (%)"].mean()
           if st.session_state.history else d["healthy"])
    c2.markdown(card("Indice de vigueur", vigor(d["healthy"], avg)), unsafe_allow_html=True)
    c3.markdown(card("Risque de propagation", risk_card(risk)), unsafe_allow_html=True)


def diagnosis_page(source):
    st.markdown('<div class="title">1. Image de la plante</div>', unsafe_allow_html=True)
    images = []
    with st.container(border=True):
        if source == "upload":
            files = st.file_uploader("Glisse une ou plusieurs photos de feuilles",
                                     type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            images = [(f.name, f.getvalue()) for f in files or []]
        else:
            shot = st.camera_input("Prends une photo de la feuille")
            if shot is not None:
                images = [("camera.jpg", shot.getvalue())]
        sig = hashlib.md5(b"".join(c for _, c in images)).hexdigest() if images else ""
        if images:
            st.caption("Sélection : " + ", ".join(n for n, _ in images))
        if st.button("Analyser la santé de la plante", type="primary",
                     use_container_width=True, disabled=not images):
            st.session_state.run_sig = sig
    if not images:
        st.info("Ajoute une photo pour lancer l'analyse.")
    elif st.session_state.get("run_sig") == sig:
        for n, c in images:
            with st.container(border=True):
                try:
                    render_result(n, c)
                except Exception as e:
                    st.error(f"Impossible d'analyser {n} : {e}")


page = st.session_state.page
hist = pd.DataFrame(st.session_state.history) if st.session_state.history else None

if page == PAGES[0]:
    st.markdown('<div class="title">Plateforme de santé des plantes</div>', unsafe_allow_html=True)
    n = 0 if hist is None else len(hist)
    ns = 0 if hist is None else int((hist["Sain"] == "Non").sum())
    cols = st.columns(4)
    for c, (t, v) in zip(cols, [("Images analysées", n), ("Feuilles saines", n - ns), ("Maladies détectées", ns),
                                ("Santé moyenne", "—" if hist is None else f"{hist['Indice de santé (%)'].mean():.0f}%")]):
        c.markdown(card(t, f'<div class="kv">{v}</div>'), unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        st.markdown("Importe ou scanne une feuille : le modèle diagnostique, localise les zones suspectes "
                    "(Grad-CAM) et propose un plan d'action.")
        b1, b2 = st.columns(2)
        b1.button("Importer une image", icon=":material/upload_file:", on_click=go, args=(PAGES[1],), type="primary", use_container_width=True)
        b2.button("Scan caméra", icon=":material/photo_camera:", on_click=go, args=(PAGES[2],), use_container_width=True)
elif page == PAGES[1]:
    diagnosis_page("upload")
elif page == PAGES[2]:
    diagnosis_page("camera")
elif page == PAGES[3]:
    st.markdown('<div class="title">Tableau de bord</div>', unsafe_allow_html=True)
    if hist is None:
        st.info("Analyse quelques images pour alimenter le tableau de bord.")
    else:
        a, b = st.columns(2, gap="large")
        with a, st.container(border=True):
            st.markdown("**Tendance de santé**")
            st.line_chart(hist["Indice de santé (%)"], height=250, color="#16a34a")
        with b, st.container(border=True):
            st.markdown("**Diagnostics fréquents**")
            st.bar_chart((hist["Plante"] + " — " + hist["État"]).value_counts(), height=250, color="#16a34a")
else:
    st.markdown('<div class="title">Historique & insights</div>', unsafe_allow_html=True)
    with st.container(border=True):
        if hist is None:
            st.info("Les diagnostics de la session apparaîtront ici.")
        else:
            st.dataframe(hist, use_container_width=True, hide_index=True)
            l, r = st.columns(2)
            l.download_button("Exporter en CSV", hist.to_csv(index=False).encode("utf-8-sig"), "diagnostics.csv", "text/csv")
            if r.button("Vider l'historique"):
                st.session_state.history, st.session_state.seen = [], set()
                st.rerun()