"""
science_validator.py — Math and science claim verification for training data quality.

Extracts mathematical claims from LLM-generated thinking/answer text and verifies
them using sympy. Designed to filter or flag scientifically incorrect conversations
before they are saved to the training dataset.

Philosophy:
  - Conservative: only flag a claim as WRONG if we're confident
  - Skips claims that can't be parsed rather than marking them as failures
  - Score = verified / (verified + failed); skipped claims don't penalise

Usage:
    from scripts_and_skills.data.science_validator import ScienceValidator

    v = ScienceValidator()

    # Validate a raw text blob (thinking or answer)
    result = v.validate_text(thinking_text)

    # Validate an OpenAI-format conversation
    result = v.validate_conversation(messages)

    if result.score < 0.7 or result.claims_failed > 0:
        # reject or flag this conversation

    # Integrate with DatasetGenerator
    result = v.validate_conversation(triplet)
    print(result.summary())
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import sympy
    from sympy import sympify, simplify, N, nsimplify
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    _SYMPY_TRANSFORMS = (
        standard_transformations
        + (implicit_multiplication_application, convert_xor)
    )
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class MathClaim:
    """A single mathematical claim extracted from text."""
    text:       str            # original matched text
    lhs:        str            # left-hand side (raw)
    rhs:        str            # right-hand side (raw)
    claim_type: str            # "equality" | "arithmetic" | "approximation"
    source:     str            # "latex" | "inline"
    verified:   Optional[bool] = None
    error:      Optional[str]  = None


@dataclass
class ValidationResult:
    """Result of validating a piece of text or a conversation."""
    score:            float        # 0.0–1.0 (1.0 = all verifiable claims correct)
    claims_found:     int          # total math claims detected
    claims_verified:  int          # claims that checked out as correct
    claims_failed:    int          # claims that were provably wrong
    claims_skipped:   int          # claims that couldn't be parsed / verified
    issues:           List[str] = field(default_factory=list)    # human-readable errors
    verified_claims:  List[str] = field(default_factory=list)    # what passed

    @property
    def has_issues(self) -> bool:
        return self.claims_failed > 0

    def summary(self) -> str:
        lines = [
            f"Score: {self.score:.2f}  "
            f"(verified={self.claims_verified}, "
            f"failed={self.claims_failed}, "
            f"skipped={self.claims_skipped})"
        ]
        for issue in self.issues:
            lines.append(f"  ✗ {issue}")
        for v in self.verified_claims:
            lines.append(f"  ✓ {v}")
        return "\n".join(lines)


# ── Preprocessing helpers ──────────────────────────────────────────────────────

_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "θ": "theta", "λ": "lamda", "μ": "mu",
    "π": "pi", "σ": "sigma", "τ": "tau", "φ": "phi", "ω": "omega",
}


def _normalize_text(s: str) -> str:
    """Normalize unicode math notation to ASCII-friendly form."""
    s = s.translate(_SUPERSCRIPT_MAP)
    for glyph, name in _GREEK.items():
        s = s.replace(glyph, name)
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    s = s.replace("≠", "!=").replace("≈", "~=").replace("≤", "<=").replace("≥", ">=")
    return s


def _to_sympy_expr(s: str) -> str:
    """Convert common math notation to something sympy can parse."""
    s = s.strip()
    # LaTeX fractions: \frac{a}{b} → (a)/(b)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    # LaTeX sqrt: \sqrt{x} → sqrt(x)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    # LaTeX \cdot → *
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    # Remove remaining LaTeX commands: \left, \right, \,, etc.
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # ^ already handled by convert_xor transform, but also handle ** form
    # Remove trailing dots / commas (artefacts from sentence parsing)
    s = s.rstrip(".,;:")
    return s.strip()


def _looks_like_math(s: str) -> bool:
    """Heuristic: does this string look like a math expression?"""
    s = s.strip()
    if not s or len(s) > 200:
        return False
    # Must have at least one digit or a common math variable
    if not re.search(r"[0-9a-zA-Z]", s):
        return False
    # Should be mostly math-safe characters
    math_chars = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                     "+-*/^()[] _.,'\\")
    ratio = sum(1 for c in s if c in math_chars) / max(len(s), 1)
    return ratio > 0.65


# ── Claim extraction ───────────────────────────────────────────────────────────

# LaTeX inline $...$  (non-greedy, no nested $)
_RE_LATEX_INLINE   = re.compile(r"\$([^$\n]{1,150})\$")
# LaTeX display $$...$$
_RE_LATEX_DISPLAY  = re.compile(r"\$\$([^$]{1,300})\$\$", re.DOTALL)
# Equality with "=" in LaTeX-extracted expressions
_RE_EQUATION_EQ    = re.compile(r"^(.+?)\s*=\s*(.+)$")
# Natural-language equality: "x = 5" or "2 + 2 = 4"
_RE_NATURAL_EQ     = re.compile(
    r"(?<![=!<>])([a-zA-Z0-9_\^+\-*/().\s]{2,60}?)\s*=\s*([a-zA-Z0-9_\^+\-*/().\s]{1,60})(?!=)"
)
# Arithmetic facts like "3 × 4 = 12" or "2 + 2 = 4"
_RE_ARITHMETIC     = re.compile(
    r"\b([0-9][0-9\s+\-*/^().]*?)\s*=\s*([0-9][0-9\s+\-*/^().]*)\b"
)


def _extract_from_latex(text: str) -> List[MathClaim]:
    """Extract equality claims from LaTeX math spans."""
    claims = []
    for pattern, label in [(_RE_LATEX_DISPLAY, "latex_display"),
                            (_RE_LATEX_INLINE,  "latex_inline")]:
        for m in pattern.finditer(text):
            raw = m.group(1).strip()
            eq = _RE_EQUATION_EQ.match(raw)
            if eq:
                lhs, rhs = eq.group(1).strip(), eq.group(2).strip()
                if _looks_like_math(lhs) and _looks_like_math(rhs):
                    claims.append(MathClaim(
                        text=m.group(0), lhs=lhs, rhs=rhs,
                        claim_type="equality", source=label,
                    ))
    return claims


def _extract_arithmetic(text: str) -> List[MathClaim]:
    """Extract simple arithmetic equalities from plain text."""
    claims = []
    # strip LaTeX spans first so we don't double-count
    clean = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    clean = re.sub(r"\$.*?\$", "", clean)
    for m in _RE_ARITHMETIC.finditer(clean):
        lhs, rhs = m.group(1).strip(), m.group(2).strip()
        # Both sides must be purely numeric/arithmetic
        if (re.match(r"^[\d\s+\-*/^().]+$", lhs) and
                re.match(r"^[\d\s+\-*/^().]+$", rhs)):
            claims.append(MathClaim(
                text=m.group(0), lhs=lhs, rhs=rhs,
                claim_type="arithmetic", source="inline",
            ))
    return claims


def extract_math_claims(text: str) -> List[MathClaim]:
    """
    Extract all verifiable mathematical claims from a text blob.
    Returns a deduplicated list of MathClaim objects.
    """
    text = _normalize_text(text)
    claims: List[MathClaim] = []
    claims.extend(_extract_from_latex(text))
    claims.extend(_extract_arithmetic(text))

    # Deduplicate by (lhs, rhs) pair
    seen: set = set()
    unique = []
    for c in claims:
        key = (c.lhs.strip(), c.rhs.strip())
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ── Sympy verification ─────────────────────────────────────────────────────────

def _verify_equality_sympy(lhs_raw: str, rhs_raw: str) -> Tuple[Optional[bool], Optional[str]]:
    """
    Use sympy to check if lhs == rhs.
    Returns (True/False/None, error_or_None).
    None means "couldn't verify" (parsing failed or too complex).
    """
    if not HAS_SYMPY:
        return None, "sympy not installed"

    lhs_s = _to_sympy_expr(lhs_raw)
    rhs_s = _to_sympy_expr(rhs_raw)

    try:
        lhs_expr = parse_expr(lhs_s, transformations=_SYMPY_TRANSFORMS)
        rhs_expr = parse_expr(rhs_s, transformations=_SYMPY_TRANSFORMS)
        diff = simplify(lhs_expr - rhs_expr)
        if diff == 0:
            return True, None
        # Try numerical evaluation for expressions that don't simplify symbolically
        try:
            diff_num = complex(N(diff, 10))
            if abs(diff_num) < 1e-8:
                return True, None
            return False, f"{lhs_raw} ≠ {rhs_raw} (diff={diff_num:.4g})"
        except Exception:
            # Can't evaluate numerically either — skip
            return None, "could not evaluate numerically"
    except Exception as e:
        return None, f"parse error: {e}"


def _verify_arithmetic_numpy(lhs_raw: str, rhs_raw: str) -> Tuple[Optional[bool], Optional[str]]:
    """Fast arithmetic check via Python eval (safe — only numeric expressions)."""
    allowed = set("0123456789+-*/ ().\n")
    if not all(c in allowed for c in lhs_raw + rhs_raw):
        return None, "non-arithmetic characters"
    try:
        lhs_val = eval(lhs_raw.replace("^", "**"), {"__builtins__": {}})  # noqa: S307
        rhs_val = eval(rhs_raw.replace("^", "**"), {"__builtins__": {}})  # noqa: S307
        ok = abs(float(lhs_val) - float(rhs_val)) < 1e-9
        if not ok:
            return False, f"{lhs_raw} = {lhs_val} ≠ {rhs_raw} = {rhs_val}"
        return True, None
    except Exception as e:
        return None, str(e)


# ── Main validator ─────────────────────────────────────────────────────────────

class ScienceValidator:
    """
    Validates mathematical and scientific claims in generated training text.

    Args:
        min_score: threshold below which conversations are considered low quality
        max_claims: maximum claims to check per text (perf guard)
    """

    def __init__(self, min_score: float = 0.7, max_claims: int = 20):
        self.min_score  = min_score
        self.max_claims = max_claims
        if not HAS_SYMPY:
            logger.warning("sympy not installed — math verification limited to arithmetic. "
                           "pip install sympy")

    def validate_text(self, text: str) -> ValidationResult:
        """Extract and verify all math claims in a text blob."""
        claims = extract_math_claims(text)[: self.max_claims]
        return self._verify_claims(claims)

    def validate_conversation(self, messages: List[Dict]) -> ValidationResult:
        """
        Validate an OpenAI-format conversation.
        Checks both the thinking field (CoT reasoning) and final content.
        """
        parts = []
        for msg in messages:
            if msg.get("role") == "assistant":
                parts.append(msg.get("thinking") or "")
                parts.append(msg.get("content") or "")
        combined = "\n\n".join(p for p in parts if p)
        if not combined:
            return ValidationResult(score=1.0, claims_found=0,
                                    claims_verified=0, claims_failed=0,
                                    claims_skipped=0)
        return self.validate_text(combined)

    def is_acceptable(self, messages: List[Dict]) -> bool:
        """Quick pass/fail check — returns True if the conversation meets min_score."""
        result = self.validate_conversation(messages)
        return result.score >= self.min_score and not result.has_issues

    # ── Internal ────────────────────────────────────────────────────────────

    def _verify_claims(self, claims: List[MathClaim]) -> ValidationResult:
        verified_list: List[str] = []
        issues:        List[str] = []
        n_verified = n_failed = n_skipped = 0

        for claim in claims:
            ok, err = self._verify_claim(claim)
            if ok is True:
                n_verified += 1
                verified_list.append(f"{claim.lhs} = {claim.rhs}")
            elif ok is False:
                n_failed += 1
                issues.append(err or f"wrong: {claim.lhs} = {claim.rhs}")
                claim.verified = False
                claim.error    = err
            else:
                n_skipped += 1
                claim.verified = None

        total_checkable = n_verified + n_failed
        score = (n_verified / total_checkable) if total_checkable > 0 else 1.0

        return ValidationResult(
            score=score,
            claims_found=len(claims),
            claims_verified=n_verified,
            claims_failed=n_failed,
            claims_skipped=n_skipped,
            issues=issues,
            verified_claims=verified_list,
        )

    def _verify_claim(self, claim: MathClaim) -> Tuple[Optional[bool], Optional[str]]:
        """Verify a single MathClaim. Returns (ok, error_message)."""
        if claim.claim_type == "arithmetic":
            ok, err = _verify_arithmetic_numpy(claim.lhs, claim.rhs)
            if ok is None and HAS_SYMPY:
                ok, err = _verify_equality_sympy(claim.lhs, claim.rhs)
            return ok, err
        # equality / latex
        if HAS_SYMPY:
            return _verify_equality_sympy(claim.lhs, claim.rhs)
        # fallback: try arithmetic
        return _verify_arithmetic_numpy(claim.lhs, claim.rhs)


# ── Batch validation helper ────────────────────────────────────────────────────

def validate_dataset(dataset_name: str, min_score: float = 0.7,
                     data_root=None) -> Dict:
    """
    Run the validator over every conversation in a PromptStore dataset.
    Returns a summary dict with per-row results.

    Usage:
        from scripts_and_skills.data.science_validator import validate_dataset
        summary = validate_dataset("my-training-data", min_score=0.8)
        print(summary)
    """
    from .prompt_store import PromptStore
    import json

    store  = PromptStore(data_root)
    df     = store.load(dataset_name)
    v      = ScienceValidator(min_score=min_score)
    results = {"total": 0, "passed": 0, "failed": 0, "no_math": 0, "rows": []}

    for _, row in df.iterrows():
        msgs_json = row.get("messages", "")
        if not msgs_json:
            convs_json = row.get("conversations", "")
            if not convs_json:
                continue
            from .prompt_store import _sharegpt_to_openai
            try:
                msgs = _sharegpt_to_openai(json.loads(convs_json))
            except Exception:
                continue
        else:
            try:
                msgs = json.loads(msgs_json)
            except Exception:
                continue

        result = v.validate_conversation(msgs)
        results["total"] += 1

        row_info = {
            "id":    row.get("id", ""),
            "score": result.score,
            "failed": result.claims_failed,
            "issues": result.issues,
        }

        if result.claims_found == 0:
            results["no_math"] += 1
            row_info["status"] = "no_math"
        elif result.score >= min_score and not result.has_issues:
            results["passed"] += 1
            row_info["status"] = "pass"
        else:
            results["failed"] += 1
            row_info["status"] = "fail"

        results["rows"].append(row_info)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json, argparse

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Science/math claim validator")
    sub = parser.add_subparsers(dest="cmd")

    p_text = sub.add_parser("text", help="Validate a string of text")
    p_text.add_argument("text", help="Text to validate (quote it)")

    p_ds = sub.add_parser("dataset", help="Validate all conversations in a dataset")
    p_ds.add_argument("dataset_name")
    p_ds.add_argument("--min-score", type=float, default=0.7)

    args = parser.parse_args()

    if args.cmd == "text":
        v = ScienceValidator()
        result = v.validate_text(args.text)
        print(result.summary())

    elif args.cmd == "dataset":
        summary = validate_dataset(args.dataset_name, args.min_score)
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
        failed_rows = [r for r in summary["rows"] if r["status"] == "fail"]
        if failed_rows:
            print(f"\nFailed rows ({len(failed_rows)}):")
            for r in failed_rows[:10]:
                print(f"  {r['id'][:8]}... score={r['score']:.2f}  issues={r['issues']}")
    else:
        parser.print_help()
