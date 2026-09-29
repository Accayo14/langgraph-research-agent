import httpx

MODELS_URL = "https://openrouter.ai/api/v1/models"
REQUIRED_PARAMS = {"tools", "structured_outputs"}


def discover_free_models(timeout: float = 10.0) -> list[str]:
    try:
        data = httpx.get(MODELS_URL, timeout=timeout).raise_for_status().json()["data"]
    except (httpx.HTTPError, KeyError, ValueError):
        return []
    free = [
        m
        for m in data
        if m["id"].endswith(":free") and REQUIRED_PARAMS <= set(m.get("supported_parameters") or [])
    ]
    free.sort(key=lambda m: m.get("context_length") or 0, reverse=True)
    return [m["id"] for m in free]
