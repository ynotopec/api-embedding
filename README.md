# api-embedding

Lance un modèle d'embeddings compatible avec vLLM et expose son API OpenAI.
Le lanceur ne suppose ni architecture, ni longueur de contexte, ni précision :
ces choix sont laissés à vLLM et aux métadonnées du modèle.

## Installation

```bash
chmod +x install.sh run.sh
./install.sh
cp .env.example .env
```

Le venv est créé avec Python 3.12 dans `~/venv/<nom-du-projet>` par défaut.
L'installation impose aussi une version récente de Transformers afin de prendre
en charge les checkpoints dont l'architecture est déclarée comme `ministral3`,
notamment `nvidia/Nemotron-3-Embed-1B-BF16`.

Pour réparer un environnement créé avec une ancienne version de Transformers,
relancer simplement l'installation :

```bash
./install.sh
```

La contrainte peut être surchargée au besoin, tout en laissant `uv` vérifier sa
compatibilité avec vLLM dans la même résolution :

```bash
TRANSFORMERS_SPEC='transformers>=5.0.0' ./install.sh
```

Ne pas limiter Transformers à une version antérieure à 5 pour un checkpoint
`ministral3` : les versions 4.x ne déclarent pas cette architecture. Le lanceur
affiche la contrainte effectivement utilisée au début de l'installation.

## Configuration minimale

`MODEL_ID` est le seul paramètre applicatif obligatoire :

```dotenv
MODEL_ID=BAAI/bge-m3
```

Il peut s'agir d'un identifiant Hugging Face ou d'un chemin local accepté par
vLLM. `API_KEY`, `MODEL_ALIAS` et `SERVED_MODEL_NAME` sont optionnels.
`SERVED_MODEL_NAME` est prioritaire sur `MODEL_ALIAS`; sans alias, vLLM expose
directement `MODEL_ID`.

Le serveur se lance ensuite avec :

```bash
./run.sh
```

L'hôte et le port peuvent être passés en arguments (valeurs par défaut :
`0.0.0.0` et `8001`) :

```bash
./run.sh 127.0.0.1 8000
```

## Options vLLM

Aucune option propre à un modèle n'est forcée. Les variables suivantes ne
sont transmises à vLLM que lorsqu'elles sont explicitement renseignées :

- `RUNNER`
- `DTYPE`
- `MAX_MODEL_LEN`
- `GPU_MEMORY_UTILIZATION`
- `MAX_NUM_SEQS`
- `MAX_NUM_BATCHED_TOKENS`
- `QUANTIZATION`
- `KV_CACHE_DTYPE`

Les drapeaux `TRUST_REMOTE_CODE`, `ENFORCE_EAGER` et
`DISABLE_LOG_REQUESTS` sont activés uniquement avec la valeur `1`. Cela permet
de conserver les valeurs par défaut de vLLM et de ne configurer que ce dont un
modèle ou un déploiement a réellement besoin.

Exemple de réglages spécifiques, si le modèle les requiert :

```dotenv
MODEL_ID=BAAI/bge-m3
RUNNER=pooling
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.10
```

## Appel OpenAI-compatible

Le champ `model` est obligatoire dans le protocole OpenAI. Sa valeur doit être
l'alias configuré, ou `MODEL_ID` lorsqu'aucun alias n'est défini.

```bash
curl -s http://127.0.0.1:8001/v1/embeddings \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BAAI/bge-m3",
    "input": ["Premier texte", "Deuxième texte"]
  }' | jq
```

Avec le client Python OpenAI :

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8001/v1",
    api_key="change-me",
)

response = client.embeddings.create(
    model="BAAI/bge-m3",
    input=["Premier texte", "Deuxième texte"],
)
print(len(response.data), len(response.data[0].embedding))
```

La route `/pooling` reste disponible lorsque le modèle et le runner vLLM la
prennent en charge. Son format dépend de la version de vLLM et du modèle.

## Variables Hugging Face

`HF_HOME` et `HUGGINGFACE_HUB_CACHE` ne sont pas définies par le lanceur :
Hugging Face utilise donc son répertoire de cache par défaut, sauf si ces
variables sont déjà configurées dans l’environnement. `HF_HUB_ENABLE_HF_TRANSFER`
et `TOKENIZERS_PARALLELISM` peuvent aussi être surchargées. Pour un modèle privé,
configurer aussi `HF_TOKEN`.
