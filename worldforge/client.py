"""client — тонкий клиент OpenRouter на stdlib (ноль зависимостей).

Как utils_azure: креды из .env рядом с репозиторием.
    OPENROUTER_API_KEY        обязателен
    OPENROUTER_MODEL          (опц.) модель по умолчанию

OpenRouter даёт единый OpenAI-совместимый эндпойнт к десяткам моделей
(x-ai/grok-*, anthropic/*, google/*, ...). Оператор выбирает модель ручкой
станка — «слепой противник» тем сильнее, чем дешевле и чужероднее замыслу.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils_azure import _load_env          # тот же загрузчик .env, без дублей

_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEF_MODEL = "x-ai/grok-2-1212"
# креды станка живут рядом с ним (worldforge/.env), не в корневом .env
_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


class OpenRouter:
    def __init__(self, model=None, env_path=None, timeout=120):
        _load_env(env_path or _ENV)
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "нет OPENROUTER_API_KEY в .env — станок без ключа не печатает. "
                "Добавь строку OPENROUTER_API_KEY=sk-or-... в .env")
        self.key = key
        self.model = model or os.environ.get("OPENROUTER_MODEL", _DEF_MODEL)
        self.timeout = timeout

    def chat(self, system, user, temperature=0.9, retries=3, model=None):
        """Один запрос → сырой текст ответа. Повтор при сетевой/5xx ошибке."""
        body = json.dumps({
            "model": model or self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            # OpenRouter просит идентификацию источника (не обязательна):
            "HTTP-Referer": "https://local/worldforge",
            "X-Title": "WorldForge",
        }
        last = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(_URL, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                last = f"HTTP {e.code}: {detail}"
                if e.code < 500 and e.code != 429:
                    break                          # 4xx (кроме 429) — не чиним
            except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
                last = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"OpenRouter не ответил за {retries} попыток: {last}")


def list_models_hint():
    """Подсказка оператору — типовые дешёвые/чужеродные модели для слепого
    противника. Не запрос к API; просто памятка ручки выбора."""
    return [
        "x-ai/grok-2-1212",
        "x-ai/grok-3-mini",
        "google/gemini-2.0-flash-001",
        "deepseek/deepseek-chat",
        "meta-llama/llama-3.3-70b-instruct",
        "mistralai/mistral-small",
    ]
