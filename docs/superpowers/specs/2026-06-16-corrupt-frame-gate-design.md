# Spec : filtre de frames corrompues dans la boucle de détection

Date : 2026-06-16
Fichier concerné : `app/detect.py`

## Problème

La caméra IP produit occasionnellement des frames corrompues : bandes
diagonales vert néon / magenta sur toute l'image (défaut de décodage ou de
capteur). Exemple : `artifacts_tmp/2026-05-28_11-41.png`.

Une telle frame diffère énormément de la frame précédente. La boucle de
détection de `detect.py` calcule un delta entre `prev` et la frame courante ;
le pic de différence dépasse le seuil et déclenche un faux mouvement (klaxon,
fenêtre vidéo, enregistrement d'une vidéo poubelle à score élevé, notification
ntfy).

Caractéristiques du défaut (confirmées) :
- transitoire : une ou quelques frames, puis l'image se rétablit seule ;
- occasionnel.

## Objectif

Détecter une frame corrompue et l'ignorer **avant** qu'elle n'entre dans le
pipeline de détection, pour qu'elle ne déclenche jamais de mouvement.

Périmètre : **boucle de détection uniquement** (`detection()`). Comme le
prébuffer est rempli dans cette même boucle, le filtre garde aussi le prébuffer
propre automatiquement. La partie « live » de `record_video` n'est pas filtrée
(décision assumée : un glitch rare dans une vidéo enregistrée est acceptable).

## Approche retenue

Filtre par anomalie de couleur. Le défaut est massivement vert néon + magenta
avec une saturation anormalement élevée ; une scène normale de garage (béton,
tons gris/mats) a très peu de pixels fortement saturés.

### Composant : `is_corrupted_frame(frame) -> bool`

Fonction pure ajoutée près des utilitaires de `detect.py`.

1. Conversion BGR -> HSV.
2. Sélection des pixels « néon » : saturation `S >= CORRUPT_SAT_MIN`,
   valeur `V >= CORRUPT_VAL_MIN`, et teinte dans les plages vert néon **ou**
   magenta.
3. Calcul de la fraction = pixels néon / pixels totaux.
4. Retourne `True` si la fraction dépasse `CORRUPT_SAT_FRAC_MAX`.

Aucun effet de bord, testable indépendamment.

Sécurité : tout le corps est protégé ; en cas d'exception inattendue la fonction
retourne `False` (fail-open). Un bug du détecteur ne doit jamais rendre la
caméra aveugle.

### Nouvelles constantes (avec les autres réglages, ~ligne 60)

- `CORRUPT_SAT_MIN` : plancher de saturation HSV pour qu'un pixel compte comme
  « néon ».
- `CORRUPT_VAL_MIN` : plancher de luminosité, pour que le bruit sombre ne
  compte pas.
- `CORRUPT_SAT_FRAC_MAX` : fraction maximale de pixels néon vert/magenta
  avant qu'une frame soit jugée corrompue.

Les trois valeurs sont fixées empiriquement (voir calibration), pas devinées.

### Intégration dans `detection()`

Juste après `frame = fetch_image(url, auth)` et la vérification `frame is None`,
**avant** `prebuffer_frames.append(...)` et avant la mise à jour de `prev` :

```python
if is_corrupted_frame(frame):
    log("Frame corrompue ignorée")
    time.sleep(delay)
    continue
```

Conséquence : la frame corrompue n'entre pas dans le prébuffer, ne devient pas
`prev`, ne déclenche rien. La frame normale suivante est comparée à la dernière
frame normale.

## Calibration (script jetable, non livré)

Petit script d'analyse servant uniquement à fixer les seuils :

1. Charge `artifacts_tmp/2026-05-28_11-41.png` en **recadrant** pour retirer
   l'interface VLC (garder seulement la zone vidéo).
2. Échantillonne des frames de plusieurs vidéos `app/captures/*.avi` (frames
   normales).
3. Calcule la fraction néon pour les deux groupes.
4. Affiche min / max / moyenne des frames normales vs la frame corrompue.

Choix du seuil : `CORRUPT_SAT_FRAC_MAX` placé dans l'écart entre le max des
frames normales et la valeur de la frame corrompue, avec marge de sécurité du
côté des frames normales. Un seul échantillon corrompu (le screenshot recadré)
suffit pour cette première version ; seuil conservateur, réajustable plus tard.

Compromis assumé : un faux négatif (un glitch passe de temps en temps) est
acceptable ; un faux positif (jeter une vraie frame) ne l'est pas.

## Tests

- Vérification type unitaire : `is_corrupted_frame` retourne `True` sur le crop
  corrompu et `False` sur chaque frame normale échantillonnée.
- Manuel : lancer en live (ou `--test`) et confirmer que « Frame corrompue
  ignorée » apparaît lors d'un glitch et que le vrai mouvement déclenche
  toujours.

## Notes

- Le projet n'est pas un dépôt git (c'est un venv, `.gitignore` = `*`) ; cette
  spec n'est donc pas committée.
