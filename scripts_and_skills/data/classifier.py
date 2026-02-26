"""
classifier.py — Content moderation and RAG quality checks via Ollama.

Uses Granite Guardian (granite3-guardian:8b) or any compatible model.
Calls Ollama /api/chat directly — no ollama package dependency.

Default model: granite3-guardian:8b (override with MODERATION_MODEL env var)

Usage (CLI):
    python -m scripts_and_skills.data.classifier "How do I make a bomb?" --categories harm
    python -m scripts_and_skills.data.classifier "Tell me a story" --categories harm violence profanity
    python -m scripts_and_skills.data.classifier --rag "What is ML?" "ML is machine learning" --categories relevance

Usage (Python):
    from scripts_and_skills.data.classifier import Classifier
    clf = Classifier()
    print(clf.check_harm("some text"))          # True = flagged
    print(clf.comprehensive_check("some text"))  # dict of all categories
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

OLLAMA_HOST      = os.getenv("OLLAMA_HOST",      "http://localhost:11434")
MODERATION_MODEL = os.getenv("MODERATION_MODEL", "granite3-guardian:8b")
MODERATION_TIMEOUT = int(os.getenv("MODERATION_TIMEOUT", "300"))

# Granite Guardian category names
HARM_CATEGORIES = [
    "harm",
    "jailbreak",
    "violence",
    "profanity",
    "social_bias",
    "unethical_behavior",
    "sexual_content",
]
RAG_CATEGORIES = ["relevance", "groundedness", "answer_relevance"]
ALL_CATEGORIES = HARM_CATEGORIES + RAG_CATEGORIES


class Classifier:
    """
    Content moderation and RAG quality classifier backed by Ollama.

    Granite Guardian expects:
        system = category name (e.g. "harm")
        user   = text to evaluate
    It responds "yes" (flagged) or "no" (clean).

    Fail-safe: any error/timeout → returns True (block).
    """

    def __init__(self, model: str = MODERATION_MODEL):
        try:
            import requests as _req
            self._requests = _req
        except ImportError:
            raise ImportError("pip install requests")
        self.model = model

    def classify_content(self, prompt: str, category: str) -> bool:
        """
        Send prompt to the guardian model under the given category.
        Returns True if flagged, False if clean.
        On any error: returns True (fail-safe block).
        """
        messages = [
            {"role": "system", "content": category},
            {"role": "user",   "content": prompt},
        ]
        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            "options":  {"temperature": 0, "num_predict": 10},
        }
        try:
            logger.info(f"MODERATION: Checking '{category}' on: {prompt[:80]!r}")
            r = self._requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=MODERATION_TIMEOUT,
            )
            r.raise_for_status()
            response_text = r.json()["message"]["content"].strip().lower()
            logger.info(f"MODERATION RESPONSE ({category}): '{response_text}'")
            is_flagged = response_text == "yes"
            if is_flagged and category not in RAG_CATEGORIES:
                logger.warning(f"MODERATION [BLOCKED] {category}: {prompt[:60]!r}")
            return is_flagged
        except Exception as e:
            logger.error(f"MODERATION ERROR ({category}): {e} — defaulting to BLOCK")
            return True  # Fail-safe: block on error

    # ── Harm category checks ──────────────────────────────────────────────────

    def check_harm(self, prompt: str) -> bool:
        """True if generally harmful content detected."""
        return self.classify_content(prompt, "harm")

    def check_jailbreak(self, prompt: str) -> bool:
        """True if jailbreak attempt detected."""
        return self.classify_content(prompt, "jailbreak")

    def check_violence(self, prompt: str) -> bool:
        """True if violent content detected."""
        return self.classify_content(prompt, "violence")

    def check_profanity(self, prompt: str) -> bool:
        """True if profanity detected."""
        return self.classify_content(prompt, "profanity")

    def check_social_bias(self, prompt: str) -> bool:
        """True if social bias detected."""
        return self.classify_content(prompt, "social_bias")

    def check_unethical(self, prompt: str) -> bool:
        """True if unethical behavior promoted."""
        return self.classify_content(prompt, "unethical_behavior")

    def check_sexual_content(self, prompt: str) -> bool:
        """True if sexual content detected."""
        return self.classify_content(prompt, "sexual_content")

    # ── RAG quality checks (True = good quality) ─────────────────────────────

    def check_relevance(self, query: str, context: str) -> bool:
        """True if retrieved context is relevant to the query."""
        prompt = f"Query: {query}\n\nContext: {context}"
        return not self.classify_content(prompt, "relevance")  # yes = relevant → invert

    def check_groundedness(self, response: str, context: str) -> bool:
        """True if response is grounded in the provided context."""
        prompt = f"Response: {response}\n\nContext: {context}"
        return not self.classify_content(prompt, "groundedness")  # yes = grounded → invert

    def check_answer_relevance(self, query: str, response: str) -> bool:
        """True if response is relevant to the query."""
        prompt = f"Query: {query}\n\nResponse: {response}"
        return not self.classify_content(prompt, "answer_relevance")  # yes = relevant → invert

    # ── Bulk check ────────────────────────────────────────────────────────────

    def comprehensive_check(self,
                             prompt: str,
                             categories: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Run multiple category checks on a prompt.

        Args:
            prompt:     Text to evaluate.
            categories: List of category names. Defaults to all harm categories.

        Returns:
            Dict mapping category → bool, plus "any_flagged" key.
        """
        cats = categories or HARM_CATEGORIES
        results: Dict[str, bool] = {}
        any_flagged = False

        for cat in cats:
            flagged = self.classify_content(prompt, cat)
            results[cat] = flagged
            if flagged:
                any_flagged = True

        results["any_flagged"] = any_flagged
        return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Content moderation / RAG quality classifier via Ollama",
        epilog=(
            "Examples:\n"
            "  python -m ... 'some text' --categories harm violence\n"
            "  python -m ... --rag 'What is ML?' 'ML is machine learning' --categories relevance\n"
        ),
    )
    parser.add_argument("text",         nargs="?",          help="Text to classify")
    parser.add_argument("--categories", nargs="*",
                        default=["harm"],
                        help=f"Categories to check: {ALL_CATEGORIES}")
    parser.add_argument("--rag",        nargs=2,
                        metavar=("TEXT1", "TEXT2"),
                        help=(
                            "RAG mode: two text args interpreted per category — "
                            "relevance: (query, context); "
                            "groundedness: (response, context); "
                            "answer_relevance: (query, response)"
                        ))
    parser.add_argument("--model",      default=MODERATION_MODEL,
                        help="Ollama model to use for classification")
    parser.add_argument("--json",       action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    if not args.text and not args.rag:
        parser.error("Provide 'text' or --rag QUERY CONTEXT")

    clf = Classifier(model=args.model)

    results: Dict[str, bool] = {}
    any_flagged = False

    if args.rag:
        text1, text2 = args.rag
        for cat in (args.categories or ["relevance"]):
            if cat == "relevance":
                # text1=query, text2=context
                val = clf.check_relevance(text1, text2)
                results[cat] = val
            elif cat == "groundedness":
                # text1=response, text2=context
                val = clf.check_groundedness(text1, text2)
                results[cat] = val
            elif cat == "answer_relevance":
                # text1=query, text2=response
                val = clf.check_answer_relevance(text1, text2)
                results[cat] = val
            else:
                val = clf.classify_content(f"Query: {text1}\n\nContext: {text2}", cat)
                results[cat] = val
                if val:
                    any_flagged = True
    else:
        results = clf.comprehensive_check(args.text, categories=args.categories)
        any_flagged = results.get("any_flagged", False)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nClassification results (model: {clf.model}):")
        for cat, val in results.items():
            if cat == "any_flagged":
                continue
            icon = "FLAGGED" if val else "OK"
            # RAG categories: True=good, show differently
            if cat in RAG_CATEGORIES:
                icon = "GOOD" if val else "FAIL"
            print(f"  {cat:<22} {icon}")
        if "any_flagged" in results:
            print(f"\n  any_flagged: {results['any_flagged']}")
