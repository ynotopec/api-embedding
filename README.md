# api-embedding

Serveur vLLM CUDA 13 pour `BAAI/bge-m3`, exposé via :

- `/v1/embeddings` : API OpenAI-compatible pour dense embeddings
- `/pooling` : API vLLM pooling, utile pour les sorties spécifiques type multi-vector

BGE-M3 :
- dimension dense : 1024
- max length : 8192 tokens
- multilingue
- dense / sparse / ColBERT côté modèle

## Installation

```bash
chmod +x install.sh run.sh
./install.sh

cp .env.example .env
nano .env
```

Le venv est créé automatiquement dans :

```bash
~/venv/<basename project dir>
```

Exemple :

```bash
/home/ailab/api-embedding
/home/ailab/venv/api-embedding
```

## Lancement

```bash
./run.sh
```

ou :

```bash
./run.sh 0.0.0.0 8001
```

## Healthcheck

```bash
curl -i http://127.0.0.1:8001/health
```

## Test OpenAI-compatible `/v1/embeddings`

```bash
curl -s http://127.0.0.1:8001/v1/embeddings \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BAAI/bge-m3",
    "input": [
      "Paris est la capitale de la France.",
      "CUDA accélère les calculs GPU."
    ],
    "encoding_format": "float"
  }' | jq '.data[0].embedding | length'
```

Résultat attendu :

```text
1024
```

Voir un extrait :

```bash
curl -s http://127.0.0.1:8001/v1/embeddings \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BAAI/bge-m3",
    "input": "Paris est la capitale de la France.",
    "encoding_format": "float"
  }' | jq '{
    model,
    object,
    dim: (.data[0].embedding | length),
    first_values: .data[0].embedding[0:5],
    usage
  }'
```

## Test avec client Python OpenAI

```bash
source ~/venv/$(basename "$PWD")/bin/activate

python - <<'PY'
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8001/v1",
    api_key="change-me",
)

resp = client.embeddings.create(
    model="BAAI/bge-m3",
    input=[
        "Paris est la capitale de la France.",
        "CUDA accélère les calculs GPU.",
    ],
    encoding_format="float",
)

print("count:", len(resp.data))
print("dim:", len(resp.data[0].embedding))
print("first:", resp.data[0].embedding[:5])
print("usage:", resp.usage)
PY
```

## Test `/pooling`

`/pooling` est l’API vLLM utile pour les modèles pooling et les usages plus spécifiques que l’API OpenAI standard.

```bash
curl -s http://127.0.0.1:8001/pooling \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BAAI/bge-m3",
    "input": [
      "Paris est la capitale de la France.",
      "CUDA accélère les calculs GPU."
    ]
  }' | jq
```

Selon la version vLLM, le schéma exact retourné par `/pooling` peut différer. Pour dense embeddings OpenAI-compatible, privilégier `/v1/embeddings`.

## Config recommandée DGX Spark

Config safe si d’autres modèles tournent déjà :

```bash
MODEL_ID=BAAI/bge-m3
GPU_MEMORY_UTILIZATION=0.10
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=64
MAX_NUM_BATCHED_TOKENS=32768
ENFORCE_EAGER=0
```

Si mémoire insuffisante :

```bash
GPU_MEMORY_UTILIZATION=0.08
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=32
MAX_NUM_BATCHED_TOKENS=16384
```

Si GPU libre et besoin de débit :

```bash
GPU_MEMORY_UTILIZATION=0.20
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=128
MAX_NUM_BATCHED_TOKENS=65536
```

## Sparse / ColBERT

BGE-M3 sait produire dense, sparse et ColBERT, mais l’API OpenAI `/v1/embeddings` expose surtout le dense embedding classique.

Pour LightRAG/OpenWebUI classique :

```text
embedding endpoint = http://host:8001/v1/embeddings
model = BAAI/bge-m3
dimension = 1024
```

Pour sparse / ColBERT, il faut vérifier le schéma `/pooling` supporté par ta version vLLM ou utiliser directement FlagEmbedding si tu veux explicitement récupérer :

- `dense_vecs`
- `lexical_weights`
- `colbert_vecs`

## Voir la VRAM

```bash
nvidia-smi
```

ou :

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv
```

## systemd

Adapter `User`, `WorkingDirectory`, `VENV_DIR`, et le port si besoin.

```bash
sudo cp systemd/api-embedding.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now api-embedding
sudo journalctl -u api-embedding -f
```

[1]: https://docs.vllm.ai/en/v0.12.0/examples/online_serving/pooling/ "Pooling models - vLLM"
[2]: https://huggingface.co/BAAI/bge-m3 "BAAI/bge-m3 · Hugging Face"
