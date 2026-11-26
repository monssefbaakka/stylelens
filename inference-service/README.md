# inference-service — Service IA (FastAPI)

**Statut : prêt (phase P5).**

Reçoit une image et renvoie son embedding, calculé par le même code que celui qui a indexé le catalogue (`ml/stylelens/embeddings.py` : pad-to-square sur fond blanc, redimensionnement, normalisation ImageNet, CNN, normalisation L2). Un vecteur de requête et un vecteur du catalogue sont donc directement comparables.

## Endpoints

| Méthode | Chemin | Rôle |
|---------|--------|------|
| `GET` | `/health` | état, modèle chargé (`model`, `fine_tuned`, `dim`, `image_size`, `device`) |
| `POST` | `/embed` | image (multipart, champ `file`) → `{embedding, dim, model, fine_tuned, elapsed_ms}` |
| `GET` | `/docs` | documentation OpenAPI interactive |

Erreurs : `400` image illisible ou trop grande en pixels, `413` fichier trop lourd, `415` type non accepté, `422` champ `file` absent.

Traitement des photos utilisateur, avant le prétraitement du catalogue :

- orientation EXIF appliquée (photos de téléphone) ;
- PNG / WebP transparents aplatis sur fond blanc, comme les photos du catalogue ;
- taille vérifiée sur l'en-tête avant le décodage (protection contre les « bombes » de décompression).

## Configuration (variables d'environnement)

| Variable | Défaut | Rôle |
|----------|--------|------|
| `MODEL_CHECKPOINT` | — | modèle fine-tuné (`ml/models/<backbone>-ft/model.pt`, P3) ; le service refuse de démarrer si le fichier manque |
| `MODEL_NAME` | `resnet50` | backbone ImageNet si pas de checkpoint (`efficientnet_b0`, `mobilenet_v3_large`) |
| `MODEL_PRETRAINED` | `true` | `false` : poids aléatoires (tests) |
| `IMAGE_SIZE` | taille d'entraînement du checkpoint, sinon 224 | |
| `DEVICE` | automatique | `cuda`, `mps` ou `cpu` |
| `TORCH_THREADS` | PyTorch | threads CPU par inférence |
| `MAX_UPLOAD_BYTES` | 10 Mio | |
| `MAX_IMAGE_PIXELS` | 40 000 000 | |

Le modèle servi doit être celui qui a produit les embeddings importés dans PostgreSQL (`python -m stylelens.vectordb load`) : catalog-api compare `model`, `fine_tuned` et `dim` à `catalog_meta`.

## Lancement local

```bash
cd inference-service
pip install -r requirements-dev.txt          # + torch / torchvision (voir ml/requirements.txt)
MODEL_CHECKPOINT=../ml/models/resnet50-ft/model.pt uvicorn --factory app.main:create_app --port 8000
curl -F file=@photo.jpg http://localhost:8000/embed
python -m pytest                              # modèle à poids aléatoires, pas de téléchargement
```

`../ml` est ajouté au `PYTHONPATH` par `pyproject.toml` (tests) et copié dans l'image Docker.

## Docker

```bash
docker build -f inference-service/Dockerfile -t stylelens-inference .     # depuis la racine
```

PyTorch CPU. Le dossier `ml/models` est monté en lecture seule sur `/models` par `docker-compose.yml`, les poids ImageNet téléchargés sont gardés dans un volume (`/cache/torch`).
