## Build Stack

1. Create venv

```bash
$ uv venv -p <python environment> .venv
$ uv add llama-stack
```

2. Build llama stack distribution

```bash
$ uv run llama stack build --template starter --image-type venv --config llama-stack-config.yaml
```

3. Run Stack

```bash
$ uv run llama stack run --image-type venv llama-stack-config.yaml
```
