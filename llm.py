"""One small wrapper for any LLM: Anthropic, or anything OpenAI-compatible (Gemini, OpenAI, Groq...)."""
import json
import re
import time

from . import config


def extract_json(text):
    """Pull the first JSON object out of a model reply (handles ``` fences and extra text)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    text = text.replace("“", '"').replace("”", '"')
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    raise ValueError("no JSON object in model reply: " + text[:200])


class LLM:
    def __init__(self, provider=None, model=None):
        config.load_env()
        self.provider = (provider or config.env("LLM_PROVIDER", "anthropic")).lower()
        default_model = "claude-sonnet-5-5" if self.provider == "anthropic" else "gemini-2.5-flash"
        self.model = model or config.env("LLM_MODEL") or default_model
        if self.provider == "anthropic":
            import anthropic
            self.c = anthropic.Anthropic(api_key=config.env("ANTHROPIC_API_KEY") or None)
        else:
            import openai
            self.c = openai.OpenAI(api_key=config.env("OPENAI_API_KEY") or "missing",
                                   base_url=config.env("LLM_BASE_URL") or None)

    def _once(self, system, messages, max_tokens, temperature):
        if self.provider == "anthropic":
            # no temperature: newer SDK/model combinations reject it
            r = self.c.messages.create(model=self.model, system=system,
                                       messages=messages, max_tokens=max_tokens)
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        r = self.c.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + messages)
        return r.choices[0].message.content or ""

    def chat(self, system, messages, max_tokens=1500, temperature=0.2):
        last = None
        for attempt in range(4):
            try:
                return self._once(system, messages, max_tokens, temperature)
            except Exception as e:                     # retry only temporary problems
                last = e
                code = getattr(e, "status_code", None)
                temporary = code in (429, 500, 502, 503, 504) or "Connection" in type(e).__name__ \
                    or "Timeout" in type(e).__name__
                if not temporary or attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        raise last

    def json(self, system, messages, **kw):
        return extract_json(self.chat(system, messages, **kw))
