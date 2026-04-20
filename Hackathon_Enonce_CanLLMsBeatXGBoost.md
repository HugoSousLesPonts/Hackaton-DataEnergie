# 🏆 CAN LLMs BEAT XGBOOST ?
<<<<<<< HEAD
### Hackathon — Forecasting des Courbes de Chxarge ENEDIS
=======
### Hackathon — Forecasting des Courbes de Charge ENEDIS
>>>>>>> 4f9d376e708724b333fe06c48db1f7a312efc7b7
**École des Ponts ParisTech | Charif EL JAZOULI | 20 Avril 2026**

---

| | |
|---|---|
| **Intervenant** | Charif EL JAZOULI |
| **Date** | 20 Avril 2026 |
| **Durée** | 3 heures |
| **Format** | Hackathon compétitif |
| **Équipes** | 3 personnes |
| **Données** | ENEDIS Open Data + Open-Meteo |
| **Horizon de prédiction** | 14 jours demi-horaires (672 points) |

---

## 1. Contexte & Enjeux

### 1.1 ENEDIS et le réseau de distribution électrique

ENEDIS est le gestionnaire du réseau de distribution d'électricité en France. Il opère **95% du réseau de distribution national**, couvrant 35 millions de points de livraison (foyers, entreprises, industries). Son rôle est distinct de RTE (gestionnaire du réseau de *transport* haute tension) : ENEDIS s'occupe du "dernier kilomètre" — les lignes moyenne et basse tension qui alimentent directement les consommateurs finals.

Chaque demi-heure, ENEDIS reconstitue à partir des données des compteurs Linky et des profils de consommation la **courbe de charge agrégée nationale** : la puissance totale soutirée du réseau à un instant donné, exprimée en MW. C'est cette série temporelle que vous allez prédire aujourd'hui.

### 1.2 Pourquoi le forecasting de charge est-il critique ?

Le réseau électrique a une contrainte physique absolue : **à chaque instant, la production doit égaler exactement la consommation**. Il n'existe pas de "stock" d'électricité à grande échelle (les batteries ne couvrent qu'une infime fraction des besoins). Un déséquilibre entraîne une variation de fréquence (nominale à 50 Hz en Europe) qui, si elle n'est pas corrigée en quelques secondes, peut provoquer des délestages en cascade — voire un black-out.

Le forecasting de charge intervient à plusieurs horizons :

| Horizon | Utilisateur | Usage concret |
|---|---|---|
| 30 min – 4h | RTE / ENEDIS | Activation des réserves, ajustement des groupes de production |
| 24h (J+1) | Traders, fournisseurs | Achats d'énergie sur le marché EPEX SPOT day-ahead |
| 7 jours | Planification réseau | Maintenance préventive, mobilisation des moyens de pointe |
| 1 mois | Direction technique | Planification des arrêts de centrales nucléaires |

**Un écart de prévision de 1 GWh** sur la période de pointe hivernale peut coûter plusieurs dizaines de milliers d'euros en achats d'énergie d'équilibrage sur le marché intraday, où les prix sont beaucoup plus élevés qu'en day-ahead.

### 1.3 Les caractéristiques de la série temporelle ENEDIS

La courbe de charge nationale présente des structures très riches que vos modèles devront capturer :

**Saisonnalité intra-journalière (période 48 demi-heures)**
La consommation suit un profil en "double bosse" caractéristique : une montée à partir de 7h (réveil, activité tertiaire), un plateau en journée, un pic en soirée vers 19-20h (retour à domicile, cuisson, chauffage), puis une chute nocturne. Ce profil varie selon le type de journée.

**Saisonnalité hebdomadaire (période 7 jours)**
Les week-ends affichent une consommation significativement plus faible (-10 à -20%) que les jours ouvrés, avec un profil décalé : le pic du matin est plus tardif le samedi, quasi absent le dimanche.

**Saisonnalité annuelle**
La consommation est fortement thermosensible : chaque degré Celsius en dessous de 15°C ajoute environ **2 400 MW** à la consommation nationale (effet thermosensible hivernal lié au chauffage électrique, très répandu en France). En été, la climatisation crée un effet inverse mais moins prononcé.

**Effets calendaires discontinus**
Les jours fériés cassent brutalement les patterns habituels (Noël, Pâques, 1er mai...). La série de test couvre **décembre 2024**, qui contient Noël — un jour atypique majeur.

**Tendance long terme**
La sobriété énergétique et l'efficacité des bâtiments induisent une légère tendance baissière sur plusieurs années, partiellement compensée par l'électrification des usages (véhicules électriques, pompes à chaleur).

### 1.4 La question centrale du hackathon

Depuis 2023-2024, une nouvelle famille de modèles a émergé : les **modèles de fondation pour séries temporelles** (Chronos, Moirai, TimesFM). Inspirés des LLMs de NLP, ils sont pré-entraînés sur des centaines de milliers de séries temporelles issues de domaines variés, et peuvent faire des prédictions **zero-shot** — sans aucun réentraînement sur vos données.

Face à eux, **XGBoost** reste le modèle de référence industriel pour le forecasting tabulaire. Avec un bon feature engineering, il est souvent difficile à battre sur des séries présentant des patterns réguliers et bien structurés — ce qui est précisément le cas des courbes de charge.

La question n'est donc pas évidente, et c'est pour ça qu'elle est intéressante.

---

## 2. Planning de la séance

| Créneau | Phase | Contenu |
|---|---|---|
| 0:00 – 0:30 | 🚀 Lancement | Présentation du défi, règles, données, démarrage starter kit |
| 0:30 – 2:30 | ⚔️ **Hackathon** | Les équipes codent — l'intervenant circule |
| 2:30 – 2:50 | 📤 Soumissions | Chaque équipe envoie son CSV de prédictions |

> ⏱ **Conseil de gestion du temps** : faites tourner la baseline naive dans les 10 premières minutes. Vous aurez immédiatement un score de référence et un CSV valide en cas de pépin. Ensuite seulement, attaquez XGBoost puis les LLMs.

---

## 3. Données

### 3.1 Source principale — Courbes de charge ENEDIS

| Paramètre | Valeur |
|---|---|
| **URL** | [data.enedis.fr](https://data.enedis.fr) → "Données de courbes de charge agrégées (RES)" |
| **Granularité** | Demi-horaire (48 points/jour) |
| **Niveau** | National agrégé |
| **Période à télécharger** | Janvier 2021 → 31 Décembre 2024 |
| **Variable cible** | Puissance soutirée reconstituée (MW) |
| **Format** | CSV téléchargeable ou API REST |

**Comment télécharger ?**
Sur data.enedis.fr, cherchez le dataset *"Données de courbes de charge générées"*. Vous pouvez soit télécharger le CSV directement, soit utiliser l'API avec le code fourni dans le starter kit. Les colonnes peuvent varier selon la version — adaptez le parsing en conséquence.

**Ordre de grandeur des valeurs**
La consommation nationale oscille entre ~25 000 MW (nuit d'été) et ~90 000 MW (soirée hivernale froide). Une valeur aberrante en dehors de cette plage doit être questionnée.

### 3.2 Source covariable — Météo Open-Meteo

| Paramètre | Valeur |
|---|---|
| **URL** | [api.open-meteo.com](https://open-meteo.com) — Gratuit, sans clé API |
| **Variables recommandées** | Température 2m (°C), Rayonnement solaire (W/m²), Vitesse du vent (km/h) |
| **Localisation** | Moyenne pondérée : Paris (35%), Lille (25%), Lyon (20%), Marseille (20%) |
| **Granularité** | Horaire → resample 30 min par interpolation linéaire |
| **Librairie Python** | `pip install openmeteo-requests requests-cache retry-requests` |

**Pourquoi une moyenne pondérée de 4 villes ?**
La France est un pays vaste avec des climats très hétérogènes. Lille subit des hivers rigoureux qui pèsent fortement sur la consommation de chauffage électrique (très répandu dans le Nord). Paris concentre une part majeure de la population. Lyon et Marseille représentent les grands bassins de consommation du Sud. Une seule station météo (Paris) serait un proxy trop incomplet.

**La variable température est de loin la plus corrélée à la charge** — avec une relation non-linéaire : sous ~15°C, chaque degré de moins ajoute ~2 400 MW. Au-dessus de ~25°C, chaque degré de plus ajoute ~600 MW (climatisation moins répandue qu'en pays méditerranéens).

### 3.3 Split temporel — règle absolue

```
Train      :  01/01/2021  →  17/12/2024   (~35 000 demi-heures)
Test caché :  18/12/2024  →  31/12/2024   (672 demi-heures = 14 jours)
```

> ⚠️ **Ne jamais utiliser de split aléatoire sur des séries temporelles.**
> Toute fuite de données futures (data leakage) entraîne la **disqualification**.
>
> En pratique : vos features de lag doivent pointer vers des dates dans le train. Un lag de 48 sur le premier point de test (18/12 00:00) pointe vers le 17/12 00:00 — qui est bien dans le train. C'est correct.

**Pourquoi décembre 2024 comme période de test ?**
C'est une période délibérément difficile : elle contient Noël (25/12) et les fêtes de fin d'année, des jours très atypiques où la consommation dévie fortement des patterns habituels. Les modèles qui n'ont pas explicitement intégré les jours fériés seront pénalisés.

---

## 4. Les modèles

### 4.1 Principe général : deux philosophies s'affrontent

**XGBoost (approche tabulaire supervisée)**
XGBoost ne "comprend" pas les séries temporelles en tant que telles. Il voit des **lignes de features** : pour chaque demi-heure à prédire, vous lui fournissez un vecteur de caractéristiques (heure, jour, température, lags, moyennes mobiles...) et il prédit la valeur cible. Sa force vient entièrement de la qualité du feature engineering. Il est rapide, interprétable (feature importance), et redoutablement efficace quand les patterns sont stables et bien capturés par vos features.

**Modèles de fondation LLM (approche zero-shot)**
Ces modèles ont été pré-entraînés sur des centaines de milliers de séries temporelles de domaines variés (énergie, finance, météo, ventes, trafic...). Ils ont "appris" des patterns universels : saisonnalité, tendance, ruptures. Pour les utiliser, vous leur donnez simplement un **contexte** (les N dernières valeurs de votre série) et ils prédisent les H prochaines valeurs — sans voir une seule ligne de vos données d'entraînement. Pas de feature engineering, pas de réentraînement.

### 4.2 ⚡ Camp LLM — Modèles de Fondation

| Modèle | Origine | Architecture | Points forts | Limite |
|---|---|---|---|---|
| **Chronos** | Amazon (2024) | T5 encoder-decoder, valeurs quantifiées comme tokens | Prévision probabiliste (intervalles de confiance), plusieurs tailles | Lent sur CPU, contexte limité |
| **Moirai** | Salesforce (2024) | Transformer universel, entraîné sur LOTSA (27B points) | Supporte les covariables (température !), multivarié natif | Installation plus complexe |
| **TimesFM** | Google DeepMind (2024) | Decoder-only, 200M paramètres | Très rapide en inférence, bonne calibration haute fréquence | Pas de covariables natives |

**Installation :**
```bash
pip install chronos-forecasting    # Chronos
pip install uni2ts gluonts         # Moirai
pip install timesfm                # TimesFM
pip install torch                  # Requis pour les trois
```

**Ce que vous pouvez faire varier :**
- La **longueur de contexte** : testez 336 pts (1 semaine), 672 pts (2 semaines), 1344 pts (1 mois). Les résultats peuvent surprendre — plus de contexte n'est pas toujours mieux.
- Pour Chronos : le **nombre de trajectoires** Monte Carlo (num_samples) influence la qualité des intervalles de confiance.
- Pour Moirai : passez la **température** comme covariable — c'est son avantage principal sur les deux autres.

### 4.3 🌲 Camp Classique — Baseline + XGBoost

**Baseline Naive Saisonnière (à implémenter en premier)**

La baseline la plus simple : prédire que demain ressemblera à aujourd'hui (lag 48) ou à la même journée la semaine dernière (lag 48×7). Sur des séries énergétiques, cette baseline est souvent étonnamment compétitive. Elle doit être votre premier résultat — si votre modèle ne bat pas la naive, quelque chose ne va pas.

**XGBoost — le champion à battre**

Sa performance dépend entièrement de vos features. Les catégories de features à considérer :

| Catégorie | Exemples | Importance |
|---|---|---|
| **Temporelles cycliques** | sin/cos heure, sin/cos jour semaine, sin/cos mois | ⭐⭐⭐⭐⭐ |
| **Météo** | temp_c, temp², temp×heure, radiation solaire | ⭐⭐⭐⭐⭐ |
| **Lags** | lag_48 (J-1), lag_336 (S-1), lag_672 (S-2) | ⭐⭐⭐⭐ |
| **Rolling stats** | mean_48, mean_336, std_48 | ⭐⭐⭐⭐ |
| **Calendaires** | is_weekend, is_holiday, dayofyear | ⭐⭐⭐ |
| **Tendance** | numéro de semaine, trend linéaire | ⭐⭐ |

> 🔑 **Encodage cyclique — obligatoire**
> Ne passez jamais l'heure brute (0 à 23) à XGBoost. Pour lui, 23 et 0 sont très éloignés alors qu'ils sont consécutifs dans le temps. Utilisez :
> ```python
> sin_hour = np.sin(2 * np.pi * hour / 24)
> cos_hour = np.cos(2 * np.pi * hour / 24)
> ```

---

## 5. Métriques d'évaluation

### 5.1 Métrique officielle : MAE

Le classement final est basé sur le **MAE (Mean Absolute Error)** exprimé en MW.

```
MAE = (1/n) × Σ |y_i - ŷ_i|
```

Le MAE s'interprète directement en MW d'erreur moyenne sur les 672 demi-heures de test. Un MAE de 1 000 MW signifie qu'en moyenne, vos prédictions s'écartent de 1 000 MW des valeurs réelles — soit environ 1,5% de la consommation nationale moyenne.

**Pourquoi le MAE plutôt que le RMSE ?**
Le RMSE pénalise fortement les erreurs ponctuelles (pics, anomalies). Pour une évaluation globale de la performance sur 14 jours, le MAE est plus robuste et plus interprétable opérationnellement.

### 5.2 Métriques secondaires (indicatives)

| Métrique | Formule | Ce qu'elle mesure |
|---|---|---|
| **MAE (MW)** | mean\|y − ŷ\| | Erreur absolue moyenne — **métrique officielle** |
| **RMSE (MW)** | sqrt(mean(y−ŷ)²) | Sensible aux erreurs ponctuelles sur les pics |
| **MAPE (%)** | mean\|y−ŷ\|/y × 100 | Erreur relative — utile pour comparer entre séries |
| **MASE** | MAE / MAE_naive | < 1 : vous battez la naive. > 1 : vous faites pire |

> Le **MASE** (Mean Absolute Scaled Error) est particulièrement instructif : il normalise votre erreur par l'erreur de la baseline naive saisonnière. Un MASE de 0.7 signifie que votre modèle fait 30% moins d'erreur que la naive — c'est un bon résultat sur des données énergétiques.

---

## 6. Format de soumission

### 6.1 Ce que vous envoyez

Un **unique fichier CSV** par email à l'intervenant, contenant les prédictions de votre **meilleur modèle** sur les 672 demi-heures de test.

### 6.2 Nom du fichier

```
equipe_NOM_DE_VOTRE_EQUIPE_MODELE_predictions.csv
```

Exemples valides :
```
equipe_AlphaTeam_XGBoost_predictions.csv
equipe_PowerForecasters_Chronos_predictions.csv
equipe_LesOubliettes_TimesFM_predictions.csv
```

### 6.3 Structure du CSV

```csv
datetime,load_mw_pred
2024-12-18 00:00:00,52341.5
2024-12-18 00:30:00,51890.2
2024-12-18 01:00:00,51203.8
2024-12-18 01:30:00,50741.1
...
2024-12-31 23:00:00,48203.4
2024-12-31 23:30:00,47890.1
```

### 6.4 Règles strictes

- ✅ **672 lignes exactement** (14 jours × 48 demi-heures)
- ✅ Pas demi-horaire strict : `00:00`, `00:30`, `01:00`, ..., `23:30`
- ✅ Période : `2024-12-18 00:00:00` → `2024-12-31 23:30:00`
- ✅ Format datetime : `YYYY-MM-DD HH:MM:SS`
- ✅ Valeurs en MW, positives, float
- ✅ Pas de NaN
- ❌ Une seule soumission par équipe — pas de resoumission
- ❌ Indiquer dans l'objet de l'email : **nom équipe + modèle soumis**

> Le starter kit contient une fonction `export_submission()` qui **valide automatiquement** votre CSV avant export. Utilisez-la — elle vous évitera les erreurs de format.

---

## 7. Critères d'évaluation & Bonus

| Critère | Points | Description |
|---|---|---|
| MAE le plus bas | 60 pts | 1er: 60 / 2e: 45 / 3e: 30 / 4e: 20 / suivants: 10 |
| Meilleur LLM de la classe | 15 pts | L'équipe avec le meilleur MAE sur un modèle LLM |
| Meilleur XGBoost de la classe | 15 pts | L'équipe avec le meilleur MAE sur XGBoost |
| Analyse orale (2 min) | 10 pts | Expliquer pourquoi votre modèle gagne ou perd |
| **BONUS : LLM > XGBoost** | **+10 pts** | Si votre meilleur LLM bat votre XGBoost dans la soumission |

> Le **bonus LLM > XGBoost** est conçu pour vous inciter à vraiment comprendre les deux approches, pas juste à soumettre la plus facile à coder. Si votre Chronos bat votre XGBoost, vous avez fait quelque chose d'intéressant.

---

## 8. Conseils stratégiques

### 8.1 Sur la gestion du temps

La plus grande erreur dans un hackathon de 2h est de passer 1h sur l'installation d'une dépendance. Voici l'ordre recommandé :

1. **(0-10 min)** Chargez les données, faites tourner la baseline naive → vous avez un CSV valide
2. **(10-60 min)** Construisez XGBoost avec feature engineering progressif (d'abord temporel, puis météo)
3. **(60-90 min)** Installez et lancez un premier LLM (TimesFM est le plus rapide)
4. **(90-110 min)** Optimisez — features supplémentaires, second LLM, tuning du contexte
5. **(110-120 min)** Choisissez votre meilleur modèle, exportez et validez le CSV

### 8.2 Sur XGBoost

La température est la feature numéro un. Avant même de coder des lags sophistiqués, ajoutez la température et ses interactions :

```python
df['temp_c']          # température brute
df['temp_sq']         = df['temp_c'] ** 2             # relation non-linéaire
df['temp_x_hour']     = df['temp_c'] * df['sin_30min'] # interaction heure/temp
df['heating_degree']  = np.maximum(15 - df['temp_c'], 0)  # degrés-jours chauffe
df['cooling_degree']  = np.maximum(df['temp_c'] - 25, 0)  # degrés-jours clim
```

### 8.3 Sur les LLMs

- **TimesFM** est le plus simple à lancer — commencez par lui si vous manquez de temps
- **Chronos** donne des intervalles de confiance (100 trajectoires Monte Carlo) — utilisez la médiane comme prédiction ponctuelle
- **Moirai** est le seul à accepter des covariables nativement — si vous avez chargé la météo, c'est son avantage décisif
- La longueur de contexte n'est pas un hyperparamètre anodin : sur une série avec forte saisonnalité hebdomadaire, passer exactement **672 pts (2 semaines)** ou **2016 pts (6 semaines)** peut faire une différence significative

### 8.4 Sur les jours fériés

Décembre 2024 contient Noël (25 décembre). Ce jour a un profil de consommation radicalement différent d'un mercredi normal : la consommation ressemble à un dimanche d'été, avec des pics décalés et une amplitude réduite. Un XGBoost sans flag `is_holiday` va faire une grosse erreur ce jour-là.

```python
jours_feries_dec_2024 = ['2024-12-25']  # Noël
df['is_holiday'] = df.index.normalize().isin(pd.to_datetime(jours_feries_dec_2024)).astype(int)
```

---

## 9. Pour aller plus loin (si vous avez du temps)

Ces pistes peuvent vous donner un avantage compétitif :

- **Fine-tuning de Chronos** sur votre série ENEDIS (Chronos-Bolt) — permet de sortir du zero-shot
- **Ensemble** : moyenne pondérée de XGBoost + Chronos, avec des poids optimisés sur la validation
- **LightGBM** à la place de XGBoost — souvent légèrement meilleur sur les séries temporelles longues
- **Features de Fourier** : ajouter des termes sin/cos de longue période pour capturer la saisonnalité annuelle
- **Résidus** : modélisez d'abord la saisonnalité avec la naive, puis prédisez les résidus avec XGBoost

---

## 10. Références

| Ressource | Lien |
|---|---|
| Chronos (Amazon 2024) | arxiv.org/abs/2403.07815 |
| Moirai (Salesforce 2024) | arxiv.org/abs/2402.02592 |
| TimesFM (Google 2024) | arxiv.org/abs/2310.10688 |
| ENEDIS Open Data | data.enedis.fr |
| Open-Meteo API | open-meteo.com |
| Thermosensibilité RTE | rte-france.com → Bilan électrique |

---

*École des Ponts ParisTech | Charif EL JAZOULI | 20 Avril 2026*
