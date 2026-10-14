# APOE — Automatic Prompt Optimization Engine

Optimiseur de prompts minimaliste en **un seul fichier, Python standard** (aucune dépendance, aucune clé API).

## Principe

Boucle d'optimisation :

1. mesurer le prompt sur le jeu d'**entraînement** ;
2. lire ses erreurs ;
3. demander une variante au méta-optimiseur (qui ne voit jamais la validation) ;
4. ne la garder que si son score sur la **validation** progresse d'au moins `MIN_GAIN`.

Cas d'usage de démonstration : classification de sentiment d'avis techniques (positif / négatif / mitigé / neutre).

## Composants

| Composant | Rôle |
|---|---|
| `TestCase` | exemple (entrée, sortie attendue) |
| `OptimizablePrompt` | instructions + template + few-shot |
| `ModelClient` | interface unique vers le modèle (cible, juge, méta-optimiseur) |
| `Evaluator` | score moyen du juge, normalisé dans [0, 1] |
| `PromptOptimizer` | boucle d'optimisation + journal |

Bonus : few-shot par similarité avec exclusion anti-fuite du cas évalué.

## Lancer

```bash
python optimiseur.py
```

Génère `journal.json` (une entrée par itération : prompt, échecs, scores, décision).

## Utiliser un vrai LLM

Le modèle est simulé (`ModeleSimule`). Pour un vrai modèle, hériter de `ModelClient` et implémenter `target`, `judge` et `optimize`.

## Limites

- 4 cas de validation seulement (1 cas = 0,25 de score) ;
- validation proche de l'entraînement ;
- juge et modèle simulés : à refaire avec un vrai modèle et un contrôle humain sur un échantillon.
