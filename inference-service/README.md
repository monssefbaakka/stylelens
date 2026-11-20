# inference-service — Service IA (FastAPI)

**Statut : à venir (phase P5).**

Rôle : reçoit une image et renvoie son embedding calculé par le CNN (PyTorch).

Technologies prévues : Python, FastAPI, Uvicorn, PyTorch, torchvision.

Endpoints prévus :

- `GET /health` : état du service et modèle chargé
- `POST /embed` : image (multipart) en entrée, vecteur normalisé L2 en sortie
