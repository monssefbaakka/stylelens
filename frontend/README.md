# frontend — Interface (React + TypeScript)

**Statut : prêt (phase P6).**

Recherche par image : l'utilisateur dépose une photo, l'interface affiche les articles du catalogue les plus ressemblants, avec leur score de similarité, et permet de filtrer par catégorie.

React 19 · TypeScript · Vite · Vitest + Testing Library · oxlint.

## Fonctionnalités

- dépôt d'image par glisser-déposer, sélection de fichier ou collage (Ctrl+V) ; vérification locale du format (JPEG, PNG, WebP) et de la taille (10 Mo) avant envoi ;
- aperçu de l'image envoyée, grille des résultats (photo, nom, type d'article, couleur, catégorie, similarité en %) ;
- filtres combinables : catégorie (puces), sous-catégorie, type d'article, genre et couleur (listes), avec le nombre de produits de chaque valeur compte tenu des autres filtres (`GET /api/filters`) ; changer de catégorie efface la sous-catégorie et le type d'article ; bouton « Réinitialiser les filtres » ;
- nombre de résultats (12, 24, 48) ; à chaque changement, la recherche est relancée et la requête précédente annulée ;
- « Articles similaires » sur un résultat : nouvelle recherche à partir de ce produit, sans renvoyer d'image, filtres conservés ;
- états de chargement (squelettes), d'erreur (message de l'API) et résultat vide ; thème clair / sombre selon le système ; mise en page mobile.

## Lancement

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 ; /api et /images relayés vers catalog-api (VITE_API_TARGET, défaut http://localhost:8080)
npm test             # tests Vitest (API simulée)
npm run lint
npm run build        # vérification des types + build de production dans dist/
```

En production (`Dockerfile`), nginx sert `dist/` et relaie `/api` et `/images` vers `catalog-api:8080`, sur la même origine.

## Structure

```
src/
├── api.ts                    # client catalog-api (types, erreurs RFC 9457, validation locale)
├── App.tsx                   # état de la recherche, annulation des requêtes périmées
├── components/
│   ├── UploadZone.tsx        # glisser-déposer, sélection, collage
│   ├── FilterPanel.tsx       # filtres à facettes (catégorie, sous-catégorie, type, genre, couleur)
│   └── ResultsGrid.tsx       # cartes produit, squelettes de chargement
├── filters.ts                # filtres dépendants (catégorie → sous-catégorie → type)
├── labels.ts                 # libellés français (catégories, genres, couleurs), formats
└── index.css                 # thème (variables CSS clair / sombre)
```
