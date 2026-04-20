# 📖 Le Forecasting de Séries Temporelles avec les LLMs
### Du Transformer NLP aux Modèles de Fondation Temporels
**École des Ponts ParisTech | Support pédagogique — Hackathon ENEDIS**

---

## Table des matières

1. [Rappel : qu'est-ce qu'une série temporelle ?](#1-rappel--quest-ce-quune-série-temporelle-)
2. [Les approches classiques et leurs limites](#2-les-approches-classiques-et-leurs-limites)
3. [Pourquoi les LLMs peuvent forecaster des séries temporelles](#3-pourquoi-les-llms-peuvent-forecaster-des-séries-temporelles)
4. [Chronos — Amazon (2024)](#4-chronos--amazon-2024)
5. [Moirai — Salesforce (2024)](#5-moirai--salesforce-2024)
6. [TimesFM — Google DeepMind (2024)](#6-timesfm--google-deepmind-2024)
7. [Comparaison des trois modèles](#7-comparaison-des-trois-modèles)
8. [Zero-shot vs Fine-tuning vs Transfer Learning](#8-zero-shot-vs-fine-tuning-vs-transfer-learning)
9. [Ce que les LLMs capturent — et ce qu'ils ratent](#9-ce-que-les-llms-capturent--et-ce-quils-ratent)
10. [LLMs vs XGBoost — quand l'un bat l'autre](#10-llms-vs-xgboost--quand-lun-bat-lautre)
11. [Lectures recommandées](#11-lectures-recommandées)

---

## 1. Rappel : qu'est-ce qu'une série temporelle ?

Une **série temporelle** est une séquence de valeurs ordonnées dans le temps :

```
y_1, y_2, y_3, ..., y_T
```

où chaque valeur `y_t` est observée à un instant `t` avec un pas de temps régulier (ici : 30 minutes).

Le **problème de forecasting** consiste à prédire les `H` prochaines valeurs :

```
Entrée  : y_{t-C}, y_{t-C+1}, ..., y_{t}    (contexte de longueur C)
Sortie  : ŷ_{t+1}, ŷ_{t+2}, ..., ŷ_{t+H}   (horizon H)
```

Ce problème ressemble structurellement à la **modélisation du langage** :

```
NLP        : "Le chat mange une ..."  → "souris"
Séries     : [52000, 51200, 50800, ...] → [50400, 50100, ...]
```

Dans les deux cas, on prédit la suite d'une séquence à partir d'un contexte. C'est cette analogie qui a inspiré les modèles de fondation temporels.

---

## 2. Les approches classiques et leurs limites

Avant de comprendre pourquoi les LLMs sont intéressants, il faut comprendre ce que les approches classiques font bien — et mal.

### 2.1 ARIMA / SARIMA

**Principe** : modélise la série comme une combinaison linéaire de ses valeurs passées (AR), de ses erreurs passées (MA), avec différenciation pour stationnarité (I). La variante SARIMA ajoute des termes saisonniers.

```
SARIMA(p,d,q)(P,D,Q)[m]
y_t = c + φ₁y_{t-1} + ... + φ_py_{t-p} + θ₁ε_{t-1} + ... + θ_qε_{t-q} + ε_t
```

**Forces** : interprétable, bien fondé statistiquement, bons intervalles de confiance.

**Limites** :
- Suppose la **linéarité** — impossible de capturer les interactions complexes (température × heure)
- Doit être **réentraîné** pour chaque nouvelle série
- Très lent sur des séries demi-horaires avec double saisonnalité (m=48 et m=336)
- Ne peut pas utiliser de covariables exogènes facilement

### 2.2 Prophet (Meta)

**Principe** : décompose la série en tendance + saisonnalités + effets calendaires, ajustés par régression bayésienne.

```
y(t) = g(t) + s(t) + h(t) + ε_t
```
- `g(t)` : tendance (linéaire ou logistique avec points de rupture)
- `s(t)` : saisonnalités (Fourier series)
- `h(t)` : effets des jours fériés

**Forces** : intègre les jours fériés nativement, robuste aux données manquantes, interprétable.

**Limites** :
- Saisonnalités supposées **stables dans le temps** (or la thermosensibilité varie)
- Ne capture pas les **dépendances à court terme** (lag 1, lag 2)
- Performances souvent inférieures à XGBoost sur des séries haute fréquence

### 2.3 XGBoost (approche tabulaire)

**Principe** : transforme le problème de forecasting en régression supervisée. Pour chaque instant `t`, on construit un vecteur de features (lags, features temporelles, météo...) et on prédit `y_t`.

**Forces** : très performant avec un bon feature engineering, gère les non-linéarités et interactions, rapide.

**Limites** :
- **Tout repose sur le feature engineering** — si vous oubliez une feature importante, le modèle ne peut pas la deviner
- Ne généralise pas à une nouvelle série sans réentraînement complet
- N'a pas de notion intrinsèque de l'ordre temporel (les lags sont une approximation)

### 2.4 Le problème fondamental de toutes ces approches

Elles doivent être **entraînées ou calibrées sur chaque nouvelle série**. Si vous avez une nouvelle région, un nouveau compteur, ou une nouvelle variable, vous repartez de zéro. C'est coûteux en temps, en données, et en expertise.

C'est exactement ce problème que les modèles de fondation cherchent à résoudre.

---

## 3. Pourquoi les LLMs peuvent forecaster des séries temporelles

### 3.1 L'hypothèse centrale

Les LLMs (GPT, T5, LLaMA...) ont montré une capacité remarquable à **généraliser** : entraînés sur du texte, ils peuvent traduire, coder, raisonner. L'hypothèse des modèles de fondation temporels est analogue :

> **Si on entraîne un Transformer sur suffisamment de séries temporelles diverses, il apprendra des patterns universels (saisonnalité, tendance, régression à la moyenne) qui lui permettront de forecaster n'importe quelle nouvelle série sans réentraînement.**

### 3.2 L'architecture Transformer appliquée aux séries temporelles

Le Transformer (Vaswani et al., 2017) est la brique fondamentale. Rappel de son fonctionnement :

**Self-Attention**
Chaque élément de la séquence peut "regarder" tous les autres et pondérer leur importance :

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Appliqué aux séries temporelles :
- `y_t` peut attendre `y_{t-48}` (même heure hier) avec un poids élevé
- `y_t` peut attendre `y_{t-336}` (même heure la semaine dernière) avec un poids élevé
- L'attention **apprend automatiquement** quelles dépendances temporelles sont importantes

**Positional Encoding**
Dans le Transformer original, la position de chaque token est encodée via des fonctions sin/cos. Pour les séries temporelles, cet encodage doit capturer la **temporalité** (heure du jour, jour de la semaine...) — ce qui ressemble à notre feature engineering manuel, mais appris automatiquement.

**Avantage sur les RNNs/LSTMs**
Les LSTM doivent propager l'information séquentiellement : pour apprendre que `y_t` dépend de `y_{t-336}`, il faut "passer" par tous les pas intermédiaires. Le Transformer accède directement à n'importe quelle position via l'attention — les dépendances longue portée sont donc beaucoup plus faciles à apprendre.

### 3.3 Le pré-entraînement sur des corpus massifs

La clé est la **diversité** des séries d'entraînement. En apprenant sur des millions de séries issues de domaines variés :

- Séries énergétiques (consommation électrique, gaz, pétrole)
- Séries financières (prix d'actions, taux de change)
- Séries météo (température, précipitations)
- Séries de ventes (retail, e-commerce)
- Trafic web, transport, santé...

Le modèle apprend des **patterns universels** qui transcendent les domaines :
- La saisonnalité est présente partout, à des périodes différentes
- Les tendances, ruptures et anomalies suivent des structures similaires
- La régression à la moyenne est un principe général

C'est l'équivalent temporel de ce qu'ImageNet a été pour la vision par ordinateur.

---

## 4. Chronos — Amazon (2024)

> Papier : *"Chronos: Learning the Language of Time Series"* — Ansari et al., 2024

### 4.1 L'idée clé : tokeniser les valeurs numériques

Chronos résout un problème fondamental : comment utiliser un LLM textuel (T5) sur des valeurs numériques continues ?

La réponse est la **quantification** : transformer les valeurs réelles en tokens discrets, exactement comme des mots dans un vocabulaire.

**Étape 1 — Normalisation**
Pour chaque série, les valeurs sont normalisées par la médiane et la déviation absolue médiane (MAD), rendant le modèle invariant à l'échelle :

```python
y_normalized = (y - median(y)) / (MAD(y) + ε)
```

**Étape 2 — Quantification**
La plage de valeurs normalisées est divisée en `B` intervalles (bins). Chaque valeur est assignée au bin correspondant. Avec B=4096 bins, on obtient un vocabulaire discret de 4096 tokens numériques.

```
y = 52341 MW  →  normalized: 0.73  →  bin: 2987  →  token: <2987>
```

**Étape 3 — Modélisation avec T5**
La séquence de tokens est traitée exactement comme du texte par un T5 encoder-decoder. L'encodeur traite le contexte, le décodeur génère les tokens futurs un par un.

**Étape 4 — Décodage probabiliste**
Pour chaque pas futur, T5 génère une **distribution de probabilité** sur les 4096 bins. En échantillonnant cette distribution 100 fois (Monte Carlo), on obtient 100 trajectoires possibles → intervalles de confiance.

```python
forecast = chronos_pipe.predict(
    context=context,
    prediction_length=672,
    num_samples=100       # 100 trajectoires Monte Carlo
)
# forecast shape : [1, 100, 672]
y_median = np.median(forecast[0], axis=0)  # prédiction ponctuelle
y_q10    = np.percentile(forecast[0], 10, axis=0)  # borne basse IC 80%
y_q90    = np.percentile(forecast[0], 90, axis=0)  # borne haute IC 80%
```

### 4.2 Données d'entraînement

Chronos a été entraîné sur le dataset **TSMixup** compilé par Amazon, contenant ~100 000 séries temporelles réelles issues de domaines variés, augmentées par des transformations (mixup, jitter, scaling).

### 4.3 Les tailles disponibles

| Modèle | Paramètres | Vitesse CPU | Précision |
|---|---|---|---|
| chronos-t5-tiny | 8M | ⚡⚡⚡⚡ | ⭐⭐ |
| chronos-t5-mini | 20M | ⚡⚡⚡ | ⭐⭐⭐ |
| chronos-t5-small | 46M | ⚡⚡ | ⭐⭐⭐⭐ |
| chronos-t5-base | 200M | ⚡ | ⭐⭐⭐⭐ |
| chronos-t5-large | 710M | 🐢 | ⭐⭐⭐⭐⭐ |

Pour un hackathon sur CPU : **small** est le meilleur compromis.

### 4.4 Utilisation pratique

```python
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# context : les N dernières valeurs de votre série
context = torch.tensor(df_train['load_mw'].values[-672:])

forecast = pipeline.predict(
    context=context.unsqueeze(0),  # [batch=1, context_len]
    prediction_length=672,
    num_samples=100,
)
```

### 4.5 Points d'attention

- Chronos **ne voit que votre série** — pas de covariables (température, jours fériés)
- Le choix de la **longueur de contexte** est crucial : trop court = perd la saisonnalité hebdomadaire, trop long = bruit inutile
- La quantification introduit une **erreur de discrétisation** qui peut affecter les séries avec de très fortes amplitudes

---

## 5. Moirai — Salesforce (2024)

> Papier : *"Unified Training of Universal Time Series Forecasting Transformers"* — Woo et al., 2024

### 5.1 L'idée clé : un Transformer universel multi-fréquence

Moirai aborde le problème différemment de Chronos. Au lieu de tokeniser les valeurs, il travaille directement avec des **patches** (segments) de la série temporelle — une approche inspirée de PatchTST et des Vision Transformers (ViT).

**Patches adaptatives**
La série est découpée en fenêtres glissantes de taille variable. Chaque patch est projeté dans l'espace d'embedding via une projection linéaire. L'avantage : le modèle voit des "morceaux" de série plutôt que des valeurs isolées, ce qui lui permet de capturer des patterns locaux.

```
Série : [y_1, y_2, ..., y_T]
Patches de taille p=32 : [y_1:32], [y_17:48], [y_33:64], ...  (avec stride)
```

**Mixte de fréquences**
Moirai est entraîné simultanément sur des séries de fréquences très différentes : secondaire, minutaire, horaire, journalier, hebdomadaire, mensuel. Pour gérer cette hétérogénéité, il utilise des **embeddings de fréquence** qui indiquent au modèle à quelle échelle temporelle il opère.

### 5.2 LOTSA — Le corpus d'entraînement

Moirai a été entraîné sur **LOTSA** (Large-Scale Open Time Series Archive), un dataset compilé par Salesforce contenant :

- **27 milliards de points** de données temporelles
- 9 domaines : énergie, transport, météo, économie, santé, web, finance, retail, nature
- Des séries de toutes les fréquences (de la seconde à l'année)

C'est le plus grand corpus d'entraînement parmi les trois modèles — ce qui explique sa robustesse à la distribution shift.

### 5.3 Avantage clé : les covariables

Moirai supporte nativement les **features dynamiques passées** (`past_feat_dynamic_real`). C'est son principal avantage sur Chronos et TimesFM dans le contexte du hackathon :

```python
from gluonts.dataset.pandas import PandasDataset

# Avec covariable température
ds = PandasDataset(
    dict(
        target=df['load_mw'],
        past_feat_dynamic_real={
            'temperature': df['temp_c'],      # ← covariable clé !
            'radiation':   df['radiation'],
        }
    ),
    freq='30T'
)
```

En lui donnant la température, Moirai peut apprendre la relation charge/température **sans que vous ayez à la modéliser explicitement**.

### 5.4 Les tailles disponibles

| Modèle | Paramètres | Usage recommandé |
|---|---|---|
| moirai-1.1-R-small | 14M | Prototypage rapide |
| moirai-1.1-R-base | 91M | Bon compromis |
| moirai-1.1-R-large | 311M | Meilleure précision |

### 5.5 Points d'attention

- L'interface **GluonTS** est plus complexe que Chronos — comptez plus de temps d'installation
- Les covariables doivent être disponibles sur **toute la période** (train + test) — la météo future (J+1) doit être fournie, ce qui est réaliste en opérationnel (prévision météo fiable à J+1)
- Très bon sur les séries avec **distribution shift** (nouveaux patterns non vus à l'entraînement)

---

## 6. TimesFM — Google DeepMind (2024)

> Papier : *"A decoder-only foundation model for time-series forecasting"* — Das et al., 2024

### 6.1 L'idée clé : decoder-only comme GPT

TimesFM adopte l'architecture **decoder-only** — celle de GPT, LLaMA, et la majorité des LLMs modernes — plutôt que l'encoder-decoder de T5 utilisé par Chronos.

**Pourquoi decoder-only ?**
Dans un encoder-decoder, l'encodeur traite le contexte complet avant que le décodeur génère les prédictions. Dans un decoder-only, tout est traité en une seule passe avec du **masquage causal** : chaque position ne peut voir que les positions précédentes. C'est plus simple, plus scalable, et en pratique aussi performant.

**Architecture détaillée**
- 200M paramètres (taille "base" des LLMs modernes)
- Patches de taille 32 (comme Moirai) — la série est patchée avant d'entrer dans le Transformer
- Projection de patch : chaque patch de 32 valeurs → vecteur de 1280 dimensions
- 20 couches Transformer avec attention multi-têtes
- Tête de régression : prédit le patch futur de 128 valeurs en sortie

```
Input  : patches de la série de contexte
Output : patches de la série future
         (avec horizon flexible jusqu'à la longueur d'entraînement max)
```

### 6.2 Données d'entraînement

TimesFM a été entraîné sur un corpus Google interne contenant :
- **Google Trends** — séries de tendances de recherche
- **Synthetic data** — séries générées avec des processus stochastiques variés (ARIMA, ETS, séries chaotiques)
- Datasets publics standards (M4, ETT, Traffic...)

L'utilisation massive de données synthétiques est une particularité de TimesFM — elle améliore la robustesse aux distributions atypiques.

### 6.3 Prévision avec quantiles

TimesFM fournit des **quantiles de prédiction** en sortie, sans Monte Carlo :

```python
import timesfm

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend='cpu',
        horizon_len=672,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id='google/timesfm-1.0-200m-pytorch'),
)

point_forecast, quantile_forecast = tfm.forecast(
    [context_list],
    freq=[0],   # 0=haute fréquence, 1=basse fréquence
)
# quantile_forecast shape : [1, 672, 9]
# quantiles : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

### 6.4 Le paramètre `freq`

TimesFM utilise un paramètre de fréquence simplifié :

| `freq` | Usage |
|---|---|
| `0` | Haute fréquence : < 1 jour (horaire, demi-horaire...) |
| `1` | Basse fréquence : ≥ 1 jour (quotidien, hebdomadaire...) |

Pour nos données demi-horaires ENEDIS : **`freq=0`**.

### 6.5 Points d'attention

- **Le plus rapide en inférence** des trois — idéal si vous manquez de temps
- Pas de covariables natives — il ne peut pas utiliser la température directement
- Excellent sur les **séries haute fréquence** (intra-journalier) — précisément notre cas
- Les quantiles sont produits **directement** (pas de Monte Carlo) → inférence déterministe et rapide

---

## 7. Comparaison des trois modèles

| Critère | Chronos | Moirai | TimesFM |
|---|---|---|---|
| **Architecture** | T5 encoder-decoder | Transformer universel | GPT decoder-only |
| **Approche numérique** | Quantification en tokens | Patches linéaires | Patches linéaires |
| **Corpus entraînement** | ~100K séries | 27B points (LOTSA) | Google Trends + synthétique |
| **Covariables** | ❌ Non | ✅ Oui (past dynamic) | ❌ Non |
| **Prévision probabiliste** | ✅ Monte Carlo | ✅ Quantiles | ✅ Quantiles directs |
| **Vitesse CPU** | Moyenne | Lente | Rapide |
| **Facilité d'installation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Tailles disponibles** | tiny→large | small→large | 200M uniquement |
| **Longueur contexte max** | 512 (small) | 512-2048 | 512 |
| **Point fort** | IC calibrés | Covariables + robustesse | Vitesse + haute fréquence |

### Quelle longueur de contexte choisir ?

C'est l'hyperparamètre le plus impactant pour les trois modèles. Voici l'intuition :

| Contexte | Points | Ce que le modèle voit | Risque |
|---|---|---|---|
| 336 | 1 semaine | 1 cycle hebdomadaire complet | Pas assez pour la tendance |
| 672 | 2 semaines | 2 cycles, variabilité inter-semaine | Bon compromis |
| 1344 | 4 semaines | Début de saisonnalité mensuelle | Plus lent, parfois moins bon |
| 2016 | 6 semaines | Bonne couverture saisonnière | Risque de bruit historique |

> **Recommandation pratique** : commencez avec 672 (2 semaines). C'est multiple de la période hebdomadaire (48×7=336) — le modèle voit exactement 2 cycles complets.

---

## 8. Zero-shot vs Fine-tuning vs Transfer Learning

Ces trois termes sont souvent confondus. Voici les distinctions précises dans le contexte des séries temporelles.

### 8.1 Zero-shot

**Définition** : le modèle est utilisé tel quel, sans aucune adaptation aux nouvelles données. Il prédit directement sur votre série ENEDIS sans jamais l'avoir vue.

```
Pré-entraînement : séries diverses du monde entier
Utilisation      : vos données ENEDIS → prédiction directe
Données requises : aucune (juste le contexte au moment de l'inférence)
```

C'est ce que vous faites dans ce hackathon avec Chronos, Moirai, et TimesFM.

**Avantage** : déploiement immédiat, aucun coût d'entraînement.
**Limite** : le modèle ne connaît pas les spécificités de votre série (thermosensibilité de la France, jours fériés français...).

### 8.2 Few-shot

**Définition** : le modèle est adapté avec un très petit nombre d'exemples (quelques dizaines à quelques centaines de points).

```
Pré-entraînement : séries diverses
Fine-tuning      : 100-1000 points de votre série ENEDIS
Utilisation      : prédiction
```

### 8.3 Fine-tuning complet

**Définition** : le modèle pré-entraîné est réentraîné sur votre dataset complet. Tous les poids sont mis à jour.

```python
# Exemple avec Chronos-Bolt (version fine-tunable de Chronos)
from chronos import ChronosBoltPipeline

pipeline = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-small")
pipeline.train(
    train_dataset=your_enedis_dataset,
    max_steps=1000,
    learning_rate=1e-4,
)
```

**Avantage** : le modèle s'adapte aux patterns spécifiques de votre série.
**Limite** : nécessite du temps, du GPU, et suffisamment de données.

### 8.4 Transfer Learning classique

**Définition** : les couches basses du modèle (features générales) sont gelées, seules les couches hautes (features spécifiques) sont réentraînées.

**Dans le contexte des séries temporelles** : geler les premières couches du Transformer (qui ont appris les patterns universels : saisonnalité, tendance) et réentraîner seulement les dernières couches sur vos données spécifiques.

```
Analogie CV : un ResNet pré-entraîné sur ImageNet, dont on réentraîne
              seulement la dernière couche FC sur un dataset médical.
```

---

## 9. Ce que les LLMs capturent — et ce qu'ils ratent

### 9.1 Ce qu'ils capturent bien

**Les patterns universels**
Saisonnalité, tendances douces, régression à la moyenne — ces structures sont présentes dans toutes les séries du monde. Après entraînement sur des millions de séries, le modèle les reconnaît immédiatement dans votre contexte.

**Les dépendances longue portée**
Grâce à l'attention, le modèle peut directement relier `y_t` à `y_{t-336}` (même heure la semaine dernière) sans avoir à "traverser" tous les pas intermédiaires. C'est une vraie force sur les séries avec forte saisonnalité hebdomadaire.

**L'incertitude**
Chronos et Moirai fournissent des intervalles de confiance probabilistes. Un XGBoost standard ne donne pas d'intervalles (sauf avec des techniques supplémentaires comme la régression quantile).

**La robustesse aux séries courtes**
Un SARIMA a besoin de plusieurs cycles complets pour estimer ses paramètres. Un LLM zero-shot prédit dès le premier point — utile pour les nouvelles installations, les nouvelles régions.

### 9.2 Ce qu'ils ratent ou gèrent mal

**Les covariables exogènes (pour Chronos et TimesFM)**
Si la température monte de 10°C en une semaine (vague de froid soudaine), XGBoost le voit immédiatement via la feature `temp_c`. Chronos et TimesFM ne voient que la série de charge — ils ne peuvent qu'inférer indirectement que quelque chose a changé via les patterns récents.

**Les événements discontinus**
Un jour férié, une grève nationale, un événement sportif — ces ruptures ne sont pas dans le contexte récent et ne ressemblent à rien dans les données passées. XGBoost avec un flag `is_holiday` les gère bien. Les LLMs zero-shot les ratent souvent.

**Les changements de distribution (distribution shift)**
Si la composition du parc de chauffage change fortement (déploiement massif de pompes à chaleur), la relation charge/température change. Un LLM pré-entraîné sur des données antérieures n'a pas appris ce nouveau régime.

**La précision absolue sur des séries très régulières**
Sur une série très prévisible avec des patterns stables et bien capturés par un bon feature engineering XGBoost, les LLMs zero-shot peinent souvent à faire mieux. L'expertise humaine (feature engineering) reste précieuse.

---

## 10. LLMs vs XGBoost — quand l'un bat l'autre

### 10.1 Quand les LLMs gagnent

| Situation | Pourquoi les LLMs gagnent |
|---|---|
| **Peu de données historiques** | Pas besoin d'entraîner — le zero-shot fonctionne dès le 1er point |
| **Nouvelle série sans historique** | Un compteur nouvellement installé, une nouvelle région |
| **Série avec distribution shift** | Moirai est robuste aux nouveaux régimes |
| **Besoin d'intervalles de confiance** | Chronos/Moirai donnent des IC calibrés sans travail supplémentaire |
| **Prototypage rapide** | Pas de feature engineering — résultat en 5 minutes |
| **Séries avec patterns complexes non-stationnaires** | L'attention capture des dépendances non-linéaires difficiles à featuriser |

### 10.2 Quand XGBoost gagne

| Situation | Pourquoi XGBoost gagne |
|---|---|
| **Nombreuses covariables disponibles** | Température, météo, prix, indicateurs économiques → feature engineering puissant |
| **Jours fériés et événements connus** | Flags explicites impossibles à ignorer |
| **Série longue et stable** | Les patterns sont bien appris, le feature engineering optimal |
| **Contrainte de latence** | XGBoost prédit en millisecondes, les LLMs en secondes |
| **Interprétabilité requise** | Feature importance, SHAP values — les LLMs sont des boîtes noires |
| **Ressources CPU limitées** | XGBoost est ultra-léger, les LLMs nécessitent RAM et GPU idéalement |

### 10.3 La vraie leçon

En pratique industrielle, la réponse est rarement "l'un ou l'autre". Les meilleurs systèmes de forecasting énergétique utilisent des **ensembles hybrides** :

```
Prédiction finale = α × XGBoost + β × Chronos + γ × TimesFM
```

où α, β, γ sont optimisés sur un jeu de validation. XGBoost apporte la précision sur les patterns réguliers et les covariables, les LLMs apportent la robustesse et la calibration de l'incertitude.

---

## 11. Lectures recommandées

### Papiers fondateurs

| Papier | Contribution |
|---|---|
| Vaswani et al. (2017) — *Attention Is All You Need* | Architecture Transformer originale |
| Ansari et al. (2024) — *Chronos: Learning the Language of Time Series* | arxiv.org/abs/2403.07815 |
| Woo et al. (2024) — *Unified Training of Universal Time Series Forecasting Transformers* | arxiv.org/abs/2402.02592 (Moirai) |
| Das et al. (2024) — *A decoder-only foundation model for time-series forecasting* | arxiv.org/abs/2310.10688 (TimesFM) |
| Nie et al. (2023) — *A Time Series is Worth 64 Words (PatchTST)* | arxiv.org/abs/2211.14730 |

### Contexte et benchmarks

| Ressource | Contenu |
|---|---|
| GIFT-Eval (2024) | Benchmark unifié des modèles de fondation temporels |
| Monash Time Series Repository | Dataset de référence pour l'évaluation |
| RTE — Bilan électrique | Données de référence sur la consommation française |
| ENEDIS Open Data | data.enedis.fr |

### Pour aller plus loin

| Sujet | Modèle / Outil |
|---|---|
| Fine-tuning de Chronos | Chronos-Bolt (Amazon) |
| Forecasting hiérarchique | HierarchicalForecast (Nixtla) |
| LLMs + covariables | Lag-Llama, MOIRAI avec features |
| Benchmarking | GluonTS, neuralforecast (Nixtla) |
| Production / MLOps | MLflow + Airflow + monitoring de drift |

---

*École des Ponts ParisTech | Charif EL JAZOULI | 20 Avril 2026*
