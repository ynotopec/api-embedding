# BGE-M3 API — vLLM / FlagEmbedding

Serveur local pour `BAAI/bge-m3`.

## Choix backend

### `BACKEND=vllm`

Recommandé si tu veux une API OpenAI-compatible simple et rapide :

- `POST /v1/embeddings`
- compatible clients OpenAI
- embeddings denses 1024 dimensions
- idéal OpenWebUI / LightRAG / RAG dense

### `BACKEND=flagembedding`

Recommandé si tu veux exploiter BGE-M3 au maximum :

- dense embeddings
- sparse lexical weights
- ColBERT multi-vector
- endpoint `/v1/embeddings` compatible OpenAI pour le dense
- extensions optionnelles `sparse_embedding` et `colbert_vecs`

## Installation

```bash
git clone <repo> bge-m3-api
cd bge-m3-api

cp .env.example .env
nano .env

chmod +x install.sh run.sh
./install.sh
