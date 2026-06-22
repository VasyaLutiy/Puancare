#!/usr/bin/env python3
"""
utils_azure.py — тонкий интерфейс к Azure OpenAI, отдающий СРАЗУ JSON.

Берёт креды из .env (рядом со скриптом):
    AZURE_OPENAI_ENDPOINT   напр. https://<resource>.openai.azure.com/
    AZURE_API_KEY
    AZURE_MODEL             имя деплоя, напр. gpt-5.4-mini
    AZURE_API_VERSION       (опц.) по умолчанию свежая preview

Использование:
    from utils_azure import AzureJSON
    az = AzureJSON()
    data = az.ask(system="Ты判定 is-a.", user="whale vs fish",
                  schema={"is_a": "bool", "confidence": "0..1"})
    # data — это уже dict, парсить полный ответ не нужно.
"""

import os
import json
import time

_DEF_API_VERSION = "2024-12-01-preview"


def _load_env(path=None):
    """Минимальный загрузчик .env без зависимостей."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


class AzureJSON:
    """Клиент, который ВСЕГДА возвращает разобранный JSON (dict)."""

    def __init__(self, model=None, api_version=None, env_path=None):
        _load_env(env_path)
        from openai import AzureOpenAI

        endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        api_key = os.environ["AZURE_API_KEY"]
        self.model = model or os.environ.get("AZURE_MODEL", "gpt-5.4-mini")
        self.api_version = api_version or os.environ.get("AZURE_API_VERSION", _DEF_API_VERSION)

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=self.api_version,
        )

    def ask(self, system, user, schema=None, retries=3):
        """
        system, user — текст. schema (опц.) — dict {поле: тип-подсказка},
        он подмешивается в системный промпт, чтобы модель отдала ровно эти поля.
        Возвращает dict (уже json.loads). При сбое парсинга — повтор.
        """
        sys_prompt = system.rstrip()
        if schema:
            fields = ", ".join(f'"{k}": <{v}>' for k, v in schema.items())
            sys_prompt += (
                "\n\nОтвечай ТОЛЬКО валидным JSON-объектом, без markdown, без пояснений. "
                f"Ровно с полями: {{{fields}}}."
            )
        else:
            sys_prompt += "\n\nОтвечай ТОЛЬКО валидным JSON-объектом (json)."

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]

        self._last_usage = None  # сброс перед каждым вызовом
        last_err = None
        for attempt in range(retries):
            try:
                # reasoning-модели Azure капризны: не шлём temperature/max_tokens,
                # просим JSON-режим (в промпте есть слово 'json' — требование Azure).
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                self._last_usage = resp.usage  # для oracle.py — учёт токенов
                content = resp.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Azure ask failed after {retries} tries: {last_err}")


if __name__ == "__main__":
    # self-test: проверяем коннект и JSON-режим минимальным запросом
    az = AzureJSON()
    print(f"[self-test] model={az.model} api_version={az.api_version}")
    out = az.ask(
        system="Ты тестовый эхо-классификатор животных.",
        user="Кит — это рыба или млекопитающее? Дай короткий вердикт.",
        schema={"class": "fish|mammal", "confidence": "0..1", "is_fish": "bool"},
    )
    print("[self-test] получен dict:", out)
    assert isinstance(out, dict), "ответ не dict!"
    print("[self-test] OK — JSON-интерфейс работает")
