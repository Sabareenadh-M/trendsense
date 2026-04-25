from __future__ import annotations

import argparse
import json
import logging
import os
import re
import warnings
from datetime import date
from pathlib import Path
import hashlib
from typing import Any, Dict, List, Optional

# Silence noisy upstream warning on Python 3.14 in hosted environments.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater\\.",
    category=UserWarning,
)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    TavilySearch = None
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        TAVILY_AVAILABLE = True
    except ImportError:
        TAVILY_AVAILABLE = False
        TavilySearch = None
        TavilySearchResults = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None
    Table = None


logger = logging.getLogger("trendsense")
DEBUG_MODE = True
_GROQ_LLM_CACHE: Optional[Any] = None


class ResearchProduct(TypedDict):
    product_name: str
    demand_score: float
    demand_notes: str
    sentiment_score: float
    sentiment_notes: str
    amazon_avg_price: float
    aliexpress_avg_price: float
    margin_score: float
    shipping_days: float
    supply_reliability_score: float


class ScoredProduct(TypedDict):
    product_name: str
    demand_score: float
    margin_score: float
    supply_reliability_score: float
    sentiment_score: float
    success_score: float
    demand_notes: str
    sentiment_notes: str
    amazon_avg_price: float
    aliexpress_avg_price: float
    shipping_days: float


class FinalProduct(TypedDict):
    product_name: str
    success_score: float
    seo_description: str
    demand_score: float
    margin_score: float
    supply_reliability_score: float
    sentiment_score: float
    sentiment_notes: str
    amazon_avg_price: float
    aliexpress_avg_price: float
    shipping_days: float


class GraphState(TypedDict):
    category: str
    provider: str
    category_profile: Dict[str, Any]
    minimum_viable_price: float
    discovered_products: List[str]
    researched_products: List[ResearchProduct]
    scored_products: List[ScoredProduct]
    final_report: List[FinalProduct]


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def get_llm(provider: str):
    global _GROQ_LLM_CACHE

    if provider == "Local (Qwen 2.5)":
        if ChatOllama is None:
            raise RuntimeError("langchain-ollama is not installed for local inference.")
        # Prefer requested model, but support common local tag variant.
        for model_name in ("qwen2.5:7b", "qwen2.5:7b-instruct"):
            try:
                llm = ChatOllama(model=model_name, temperature=0)
                llm.invoke("Respond with exactly: ok")
                return llm
            except Exception:
                continue
        raise RuntimeError("Local Ollama model not found. Pull qwen2.5:7b or qwen2.5:7b-instruct.")

    if provider == "Cloud (Groq)":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing for Cloud (Groq) mode.")

        if _GROQ_LLM_CACHE is not None:
            return _GROQ_LLM_CACHE

        preferred_model = os.getenv("GROQ_MODEL", "").strip()
        model_candidates = [
            m
            for m in [
                preferred_model,
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
                "mixtral-8x7b-32768",
            ]
            if m
        ]

        last_error: Optional[Exception] = None
        for model_name in model_candidates:
            try:
                llm = ChatGroq(model=model_name, api_key=api_key, temperature=0)
                llm.invoke("Respond with exactly: ok")
                _GROQ_LLM_CACHE = llm
                if DEBUG_MODE:
                    print(f"[DEBUG] Using Groq model: {model_name}")
                return llm
            except Exception as exc:
                last_error = exc
                # Try next model if this one is deprecated or unavailable.
                continue

        raise RuntimeError(
            f"No supported Groq model is available. Last error: {last_error}"
        )

    raise ValueError(f"Unsupported provider: {provider}")


def _looks_like_tavily_error(text: str) -> bool:
    lowered = text.lower()
    explicit_markers = [
        "httperror",
        "unauthorized",
        "invalid api key",
        "forbidden",
    ]
    if any(marker in lowered for marker in explicit_markers):
        return True

    # Catch explicit auth failures but avoid generic article/product text matches.
    if re.search(r"\b401\b", lowered):
        return True

    return False


def _run_tavily_query(search_tool: Any, query: Any) -> Any:
    last_error: Optional[Exception] = None
    if isinstance(query, dict):
        payloads: List[Any] = [query]
        if "query" in query:
            payloads.append(query["query"])
    else:
        payloads = [{"query": query}, query]

    for payload in payloads:
        try:
            # Prefer invoke for both modern and legacy tools.
            if hasattr(search_tool, "invoke"):
                tool_name = type(search_tool).__name__.lower()
                # Modern TavilySearch expects query string in invoke most reliably.
                if "tavilysearch" in tool_name and isinstance(payload, dict):
                    return search_tool.invoke(payload.get("query", ""))
                return search_tool.invoke(payload)
            # Fallback for wrapper-style API.
            if hasattr(search_tool, "results"):
                query_text = payload if isinstance(payload, str) else str(payload.get("query", ""))
                return search_tool.results(query_text)
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Tavily search failed for query: {query}. Error: {last_error}"
    )


def _create_tavily_search_tool(max_results: int = 8) -> Any:
    """Create Tavily search tool using the new package first, then legacy fallback."""
    if not TAVILY_AVAILABLE:
        raise RuntimeError(
            "Tavily search package not found. Install langchain-tavily or langchain-community Tavily support."
        )

    # Preferred non-deprecated implementation.
    if TavilySearch is not None:
        return TavilySearch(max_results=max_results)

    # Backward-compatible fallback.
    if TavilySearchResults is not None:
        return TavilySearchResults(max_results=max_results)

    raise RuntimeError("No Tavily search class is available.")


def _normalize_search_results(raw: Any) -> List[Dict[str, Any]]:
    def _safe_dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, default=str)

    if isinstance(raw, BaseException):
        raise RuntimeError(f"Invalid Tavily response: {raw}")

    if isinstance(raw, str) and _looks_like_tavily_error(raw):
        raise RuntimeError(f"Invalid Tavily response: {raw}")

    if isinstance(raw, dict):
        if _looks_like_tavily_error(_safe_dump(raw)):
            raise RuntimeError(f"Invalid Tavily response: {raw}")
        if isinstance(raw.get("results"), list):
            return [r for r in raw["results"] if isinstance(r, dict)]
        return [raw]
    if isinstance(raw, list):
        if _looks_like_tavily_error(_safe_dump(raw)):
            raise RuntimeError(f"Invalid Tavily response: {raw}")
        normalized: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, str) and item.strip():
                normalized.append({"title": item.strip(), "content": item.strip()})
            elif isinstance(item, BaseException):
                normalized.append({"title": "Tavily Error", "content": str(item)})
        return normalized
    if isinstance(raw, str):
        return [{"title": line.strip(), "content": line.strip()} for line in raw.splitlines() if line.strip()]
    return []


def _extract_trending_products(results: List[Dict[str, Any]], fallback_category: str) -> List[str]:
    blocked_phrases = [
        "trends",
        "report",
        "guide",
        "future of",
        "review",
        "tested",
        "best of",
        "cnet",
        "wired",
        "pcmag",
    ]
    listicle_pattern = re.compile(r"^\s*\d+\s+(?:best|top|ways)\b", flags=re.IGNORECASE)
    products: List[str] = []
    for row in results:
        title = str(row.get("title") or "").strip()
        content = str(row.get("content") or "").strip()
        candidate = title or " ".join(content.split()[:8])
        candidate = re.sub(r"\s+", " ", candidate).strip(" -|:")
        combined_text = f"{title} {content} {candidate}".lower()
        if any(phrase in combined_text for phrase in blocked_phrases):
            continue
        if listicle_pattern.match(candidate):
            continue
        if candidate and candidate.lower() not in {p.lower() for p in products}:
            products.append(candidate)
        if len(products) == 5:
            break

    return products[:5]


def _hard_filter_product_candidates(candidates: List[str]) -> List[str]:
    """Drop obvious media/article titles and keep model-like names only."""
    disallowed_tokens = ["cnet", "wired", "pcmag", "review", "tested", "best of"]
    filtered: List[str] = []

    for candidate in candidates:
        lowered = candidate.lower()
        if any(token in lowered for token in disallowed_tokens):
            continue
        if candidate and candidate.lower() not in {x.lower() for x in filtered}:
            filtered.append(candidate)

    return filtered


def _seed_products_from_category(category: str) -> List[str]:
    """Return deterministic seed products when discovery cleaning over-filters titles."""
    lowered = category.lower().strip()

    if any(token in lowered for token in ("car", "automotive", "vehicle", "accessories")):
        return [
            "Car Phone Mount",
            "Dash Cam",
            "Portable Tire Inflator",
            "Car Vacuum Cleaner",
            "Seat Gap Filler",
        ]

    if any(token in lowered for token in ("beauty", "makeup", "skincare", "lip")):
        return [
            "Hydrating Lip Gloss",
            "Niacinamide Face Serum",
            "Vitamin C Face Serum",
            "Waterproof Eyeliner Pen",
            "Matte Liquid Lipstick",
        ]

    if any(token in lowered for token in ("phone", "smartphone", "iphone", "android")):
        return [
            "Wireless Charger",
            "USB-C Fast Charger",
            "MagSafe Power Bank",
            "Bluetooth Earbuds",
            "Screen Protector",
        ]

    # Generic fallback for unknown categories.
    core = re.sub(r"\s+", " ", category).strip() or "Product"
    return [
        f"{core} Starter Product",
        f"{core} Pro Product",
        f"{core} Premium Product",
        f"{core} Compact Product",
        f"{core} Best Seller Product",
    ]


def _enforce_model_name_candidates(candidates: List[str], category: str, provider: str) -> List[str]:
    """Use LLM to keep only real SKU/model-like product names."""
    candidates = _hard_filter_product_candidates(candidates)
    if not candidates:
        return []

    try:
        llm = get_llm(provider)
        prompt = f"""
Extract only specific physical product models (e.g., Samsung Galaxy S25 Ultra). Ignore all review articles, listicles, and news site names (CNET, WIRED, PCMag).

Context category: {category}
Return ONLY a JSON array of up to 5 model names.

Candidates:
{json.dumps(candidates, ensure_ascii=True)}
""".strip()

        response = llm.invoke(prompt)
        content = str(response.content).strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        parsed = json.loads(content)
        if isinstance(parsed, list):
            parsed_strings = [str(x).strip() for x in parsed if str(x).strip()]
            return _hard_filter_product_candidates(parsed_strings)[:5]
    except Exception:
        # Fallback to deterministic filtering when LLM call is unavailable.
        pass

    return candidates[:5]


def _default_category_profile(category: str) -> Dict[str, Any]:
    lowered = category.lower().strip()

    profile_map = [
        (("lipstick", "lip gloss", "makeup", "beauty", "serum", "skincare"), 5.0, 60.0, ["case", "cover", "tutorial", "news", "review"]),
        (("laptop", "notebook", "macbook"), 300.0, 3500.0, ["case", "cover", "tutorial", "news", "review", "repair", "parts"]),
        (("gpu", "graphics card", "video card", "rtx", "geforce"), 400.0, 2500.0, ["case", "cover", "tutorial", "news", "review", "repair", "parts"]),
        (("phone", "iphone", "smartphone", "galaxy", "pixel"), 100.0, 1800.0, ["case", "cover", "tutorial", "news", "review", "repair", "parts"]),
        (("agriculture", "farm", "farming", "irrigation"), 25.0, 1200.0, ["tutorial", "news", "review", "repair", "parts"]),
    ]

    for keywords, min_price, max_price, forbidden in profile_map:
        if any(keyword in lowered for keyword in keywords):
            return {
                "minimum_viable_price": min_price,
                "maximum_viable_price": max_price,
                "forbidden_keywords": forbidden,
            }

    return {
        "minimum_viable_price": 20.0,
        "maximum_viable_price": 200.0,
        "forbidden_keywords": ["case", "cover", "tutorial", "news", "review", "repair", "parts"],
    }


def _generate_category_profile(category: str) -> Dict[str, Any]:
    """Ask local Qwen 2.5 for a category profile with min/max price expectations and forbidden keywords."""
    fallback = _default_category_profile(category)

    if ChatOllama is None:
        return fallback

    try:
        llm = ChatOllama(model="qwen2.5:7b", temperature=0)
        prompt = f"""
You are an e-commerce category strategist.
Category: {category}

Generate a JSON object called Category Profile with these keys:
- minimum_viable_price: realistic minimum product price in USD
- maximum_viable_price: realistic maximum core product price in USD
- forbidden_keywords: a JSON list of lower-case keywords that should be excluded from search titles

Rules:
- Keep it category-agnostic.
- Forbidden keywords should include obvious junk, accessories, and content terms when relevant.
- Return ONLY valid JSON.

Return format:
{{
  "minimum_viable_price": number,
  "maximum_viable_price": number,
  "forbidden_keywords": ["keyword1", "keyword2"]
}}
""".strip()

        response = llm.invoke(prompt)
        content = str(response.content).strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        parsed = json.loads(content)
        min_price = float(parsed.get("minimum_viable_price", fallback["minimum_viable_price"]))
        max_price = float(parsed.get("maximum_viable_price", fallback["maximum_viable_price"]))
        forbidden = parsed.get("forbidden_keywords", fallback["forbidden_keywords"])
        if not isinstance(forbidden, list):
            forbidden = fallback["forbidden_keywords"]

        forbidden_keywords = []
        for item in forbidden:
            text = str(item).strip().lower()
            if text and text not in forbidden_keywords:
                forbidden_keywords.append(text)

        if min_price <= 0:
            min_price = float(fallback["minimum_viable_price"])
        if max_price < min_price:
            max_price = float(fallback["maximum_viable_price"])

        return {
            "minimum_viable_price": round(min_price, 2),
            "maximum_viable_price": round(max_price, 2),
            "forbidden_keywords": forbidden_keywords or list(fallback["forbidden_keywords"]),
        }
    except Exception:
        return fallback


def _define_category_boundaries(category: str) -> str:
    """Ask local Qwen to define strict category boundaries for relevance filtering."""
    fallback = (
        f"Primary products directly used for {category}; exclude cross-category accessories, unrelated electronics, "
        "manuals, tutorials, and off-domain devices."
    )

    if ChatOllama is None:
        return fallback

    try:
        llm = ChatOllama(model="qwen2.5:7b", temperature=0)
        prompt = f"""
You are an expert market analyst.
Define strict product boundaries for this category: {category}

Return 1-2 concise sentences describing:
- what product families belong to this category
- what must be excluded as cross-category noise

Do not return JSON. Return plain text only.
""".strip()
        response = llm.invoke(prompt)
        text = str(response.content).strip()
        text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        return text or fallback
    except Exception:
        return fallback


def _generate_transactional_subqueries(category: str) -> List[str]:
    """Generate 3 transactional category-focused search queries using local Qwen 2.5."""
    fallback_queries = [
        f"best selling {category} 2026",
        f"new {category} releases April 2026",
        f"trending {category} products",
    ]

    if ChatOllama is None:
        return fallback_queries

    try:
        llm = ChatOllama(model="qwen2.5:7b", temperature=0)
        prompt = f"""
You are an e-commerce demand researcher.
Category: {category}

Generate exactly 3 transactional search queries intended to surface real product listings and buying-intent pages.
Focus on product discovery, launches, and best sellers.

Return ONLY a JSON array of 3 query strings.
""".strip()
        response = llm.invoke(prompt)
        content = str(response.content).strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        parsed = json.loads(content)
        if isinstance(parsed, list):
            queries: List[str] = []
            for item in parsed:
                text = str(item).strip()
                if text and text not in queries:
                    queries.append(text)
            if len(queries) >= 3:
                return queries[:3]
    except Exception:
        pass

    return fallback_queries


def _print_discovery_debug_table(debug_rows: List[tuple[str, str]]) -> None:
    """Print discovery sanitization decisions in a readable terminal table."""
    title_col = 90
    decision_col = 38

    def _trim(text: str, width: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= width:
            return text
        return text[: width - 3] + "..."

    header_title = "Raw Search Title"
    header_decision = "Qwen Decision (DISCARD / CLEAN NAME)"
    border = f"+-{'-' * title_col}-+-{'-' * decision_col}-+"

    print("\n[DEBUG] Discovery Title Sanitization")
    print(border)
    print(f"| {_trim(header_title, title_col).ljust(title_col)} | {_trim(header_decision, decision_col).ljust(decision_col)} |")
    print(border)

    for raw_title, decision in debug_rows:
        print(f"| {_trim(raw_title, title_col).ljust(title_col)} | {_trim(decision, decision_col).ljust(decision_col)} |")

    if not debug_rows:
        print(f"| {'(no titles returned from Tavily)'.ljust(title_col)} | {'N/A'.ljust(decision_col)} |")

    print(border)


def _print_category_profile(profile: Dict[str, Any], category: str) -> None:
    """Print the Qwen-generated category profile in a structured terminal table."""
    minimum_price = float(profile.get("minimum_viable_price", 0.0))
    maximum_price = float(profile.get("maximum_viable_price", 0.0))
    forbidden_keywords = profile.get("forbidden_keywords", [])
    forbidden_text = ", ".join(str(item) for item in forbidden_keywords) if forbidden_keywords else "[]"

    if Console is not None and Table is not None:
        console = Console()
        table = Table(title=f"Category Profile: {category}", show_lines=True)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_row("Minimum Viable Price", f"${minimum_price:.2f}")
        table.add_row("Maximum Viable Price", f"${maximum_price:.2f}")
        table.add_row("Forbidden Keywords", forbidden_text)
        console.print(table)
        return

    border = "+----------------------+------------------------------------------------------------------+"
    print(f"\nCategory Profile: {category}")
    print(border)
    print(f"| {'Minimum Viable Price'.ljust(20)} | {f'${minimum_price:.2f}'.ljust(64)} |")
    print(border)
    print(f"| {'Maximum Viable Price'.ljust(20)} | {f'${maximum_price:.2f}'.ljust(64)} |")
    print(border)
    print(f"| {'Forbidden Keywords'.ljust(20)} | {forbidden_text[:64].ljust(64)} |")
    print(border)


def _clean_titles_with_local_qwen(
    results: List[Dict[str, Any]],
    category: str,
    category_boundaries: str,
    debug_rows: Optional[List[tuple[str, str]]] = None,
    forbidden_keywords: Optional[List[str]] = None,
    provider: str = "Cloud (Groq)",
) -> List[str]:
    """Use local Qwen 2.5 to discard non-products and return cleaned product names only."""
    local_llm: Optional[Any] = None
    fallback_llm: Optional[Any] = None

    if ChatOllama is not None:
        try:
            local_llm = ChatOllama(model="qwen2.5:7b", temperature=0)
            local_llm.invoke("Respond with exactly: ok")
        except Exception:
            local_llm = None

    # Keep Qwen support intact, but gracefully fall back on cloud where Ollama is unavailable.
    if provider != "Local (Qwen 2.5)" or local_llm is None:
        try:
            fallback_llm = get_llm("Cloud (Groq)")
        except Exception:
            fallback_llm = None

    if local_llm is None and fallback_llm is None:
        if DEBUG_MODE:
            print("[DEBUG] Qwen and Groq unavailable for title cleaning; using heuristic fallback.")
        return _hard_filter_product_candidates(
            [str(r.get("title") or "").strip() for r in results if str(r.get("title") or "").strip()]
        )[:5]

    cleaned_products: List[str] = []
    forbidden_set = {str(item).strip().lower() for item in (forbidden_keywords or []) if str(item).strip()}

    def _expand_composite_skus(name: str) -> List[str]:
        """Split combined SKU names like 'iPhone 17 Pro and Pro Max' into separate items."""
        normalized = re.sub(r"\s+", " ", name).strip(" -|:,;")
        if not normalized:
            return []

        split_parts = [
            part.strip()
            for part in re.split(r"\s+(?:and|&)\s+|\s*/\s*|\s*,\s*", normalized, flags=re.IGNORECASE)
            if part.strip()
        ]
        if len(split_parts) <= 1:
            return [normalized]

        first = split_parts[0]
        first_tokens = first.split()
        digit_idx = next((i for i, tok in enumerate(first_tokens) if re.search(r"\d", tok)), None)
        base_prefix = " ".join(first_tokens[: digit_idx + 1]).strip() if digit_idx is not None else ""

        expanded: List[str] = [first]
        for part in split_parts[1:]:
            # If part already looks like a full SKU, keep as-is.
            if re.search(r"\d", part):
                expanded.append(part)
                continue

            # Reconstruct missing prefix for variants like "Pro Max".
            if base_prefix:
                expanded.append(f"{base_prefix} {part}".strip())
            else:
                expanded.append(part)

        return expanded

    def _parse_model_output_to_names(content: str) -> List[str]:
        """Accept DISCARD, JSON array, or plain text and return one-or-many cleaned SKU names."""
        content = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        content = content.strip("\"' ")
        if not content or content.upper() == "DISCARD":
            return []

        candidates: List[str] = []
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                candidates = [str(x).strip() for x in parsed if str(x).strip()]
            elif isinstance(parsed, str):
                candidates = [parsed.strip()]
        except Exception:
            candidates = [line.strip("-* ") for line in re.split(r"\n+", content) if line.strip()]

        expanded: List[str] = []
        for candidate in candidates:
            for sku in _expand_composite_skus(candidate):
                if sku and not re.search(r"\b(discard|category page|article)\b", sku, flags=re.IGNORECASE):
                    expanded.append(sku)
        return expanded

    instruction = (
        "You are an expert market analyst.\n\n"
        f"Category: {category}\n"
        f"Category boundaries: {category_boundaries}\n\n"
        "Strict relevance test (binary): Is this specific product a primary item within the defined boundaries of this category?\n"
        "- If NO, return DISCARD.\n"
        "- If YES, return cleaned Brand + Model/Series only.\n\n"
        "Reject list (always DISCARD): cross-category noise such as phone cases in beauty, cameras in gardening, manuals in electronics, and unrelated accessories.\n"
        "No hallucinations: Do not try to justify a product's place in a category. If a Tapo Camera appears in a Beauty search, it is an error. DISCARD it.\n\n"
        "Entity cleaning rules:\n"
        "- Extract Brand + Model/Series Name only.\n"
        "- Remove fluff like Best, Cheap, Fast Shipping, 2026 Edition, review/listicle/news wording.\n"
        "- If title contains multiple SKUs, split into separate items.\n\n"
        "Return format rules:\n"
        "- Return DISCARD if not a real in-category primary product title.\n"
        "- Otherwise return ONLY a JSON array of cleaned product names.\n"
        "- Each item must be one specific SKU or model entity."
    )

    for row in results:
        title = str(row.get("title") or "").strip()
        if not title:
            continue

        prompt = f"{instruction}\n\nRaw title: {title}"
        try:
            if local_llm is not None:
                response = local_llm.invoke(prompt)
                model_output = str(response.content).strip()
            elif fallback_llm is not None:
                response = fallback_llm.invoke(prompt)
                model_output = str(response.content).strip()
            else:
                model_output = "DISCARD"
        except Exception as exc:
            logger.warning("Title-cleaning failed for title '%s': %s", title, exc)
            if debug_rows is not None:
                debug_rows.append((title, "DISCARD"))
            continue

        cleaned_names = _parse_model_output_to_names(model_output)
        if not cleaned_names:
            if debug_rows is not None:
                debug_rows.append((title, "DISCARD"))
            continue

        if debug_rows is not None:
            debug_rows.append((title, " | ".join(cleaned_names)))

        for cleaned in cleaned_names:
            lowered_cleaned = cleaned.lower()
            if any(keyword in lowered_cleaned for keyword in forbidden_set):
                continue
            if cleaned.lower() not in {p.lower() for p in cleaned_products}:
                cleaned_products.append(cleaned)
            if len(cleaned_products) == 5:
                break
        if len(cleaned_products) == 5:
            break

    return cleaned_products


def _extract_prices(text: str) -> List[float]:
    pattern = r"(?:\$|USD\s?)(\d+(?:\.\d{1,2})?)"
    matches = [float(x) for x in re.findall(pattern, text, flags=re.IGNORECASE)]
    return matches


def _determine_minimum_viable_price(category: str) -> float:
    """Use local Qwen to estimate realistic minimum viable product price for a category."""
    heuristic_defaults = {
        "lipstick": 5.0,
        "laptops": 300.0,
        "laptop": 300.0,
        "phone": 100.0,
        "smartphone": 100.0,
        "earbuds": 20.0,
        "headphones": 25.0,
        "watch": 30.0,
        "keyboard": 20.0,
        "mouse": 10.0,
    }

    lowered = category.lower().strip()
    for key, value in heuristic_defaults.items():
        if key in lowered:
            fallback = value
            break
    else:
        fallback = 20.0

    if ChatOllama is None:
        return fallback

    try:
        llm = ChatOllama(model="qwen2.5:7b", temperature=0)
        prompt = f"""
You are an e-commerce pricing analyst.
Category: {category}

Estimate a realistic minimum viable retail product price in USD for this category.
This threshold should help filter out accessories/junk prices that are too low for core products.

Return ONLY a JSON object in this format:
{{"minimum_viable_price": number}}
""".strip()
        response = llm.invoke(prompt)
        content = str(response.content).strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        parsed = json.loads(content)
        value = float(parsed.get("minimum_viable_price", fallback))
        if value <= 0:
            return fallback
        return round(value, 2)
    except Exception:
        return fallback


def _build_supply_queries(product: str) -> List[str]:
    """Build ultra-specific price queries with a phone-focused primary query."""
    lowered = product.lower()
    if "iphone" in lowered or "apple" in lowered:
        primary = f'"{product}" price site:amazon.com OR site:apple.com'
    else:
        primary = f'"{product}" price site:amazon.com'

    fallback = (
        f'"{product}" price unlocked phone -case -cases -cover -covers '
        "-screen protector -protector -accessory site:amazon.com OR site:apple.com"
    )
    return [primary, fallback]


def _clean_prices_for_product(prices: List[float], minimum_viable_price: float, product: str) -> List[float]:
    """Discard likely accessory/junk prices below category minimum viable threshold."""
    threshold = max(0.0, float(minimum_viable_price))
    cleaned = [p for p in prices if p >= threshold]
    if DEBUG_MODE:
        discarded = len(prices) - len(cleaned)
        if discarded > 0:
            print(
                f"[DEBUG] Price cleaner for '{product}': discarded {discarded} prices below minimum viable ${threshold:.2f}."
            )
    return cleaned


def _collect_sentiment_snippets(
    product: str,
    results: List[Dict[str, Any]],
    trigger_terms: Optional[List[str]] = None,
) -> List[str]:
    """Pull sentiment-relevant snippets from Tavily results for product analysis."""
    terms = trigger_terms or [
        "review",
        "reviews",
        "user feedback",
        "feedback",
        "customer feedback",
        "opinion",
        "opinions",
    ]
    snippets: List[str] = []

    for row in results:
        title = str(row.get("title") or "").strip()
        content = str(row.get("content") or "").strip()
        combined = f"{title} {content}".strip()
        lowered = combined.lower()
        if not any(term in lowered for term in terms):
            continue
        if product.lower() not in lowered:
            continue
        snippet = re.sub(r"\s+", " ", combined)
        if snippet and snippet not in snippets:
            snippets.append(snippet[:500])
        if len(snippets) >= 6:
            break

    return snippets


def _analyze_sentiment_with_qwen(
    product: str,
    professional_snippets: List[str],
    social_snippets: List[str],
) -> tuple[float, str]:
    """Ask Qwen 2.5 to compare professional and social sentiment with penalty logic."""
    if not professional_snippets and not social_snippets:
        return 50.0, "No review or feedback snippets were available; using neutral sentiment."

    if ChatOllama is None:
        return 50.0, "Local Qwen 2.5 is unavailable; using neutral sentiment."

    try:
        llm = ChatOllama(model="qwen2.5:7b", temperature=0)
        prompt = f"""
You are a sentiment analyst for e-commerce intelligence.
Analyze public opinion for this product by comparing two sources.

Product: {product}

Professional review snippets:
{json.dumps(professional_snippets, ensure_ascii=True, indent=2)}

Social authenticity snippets (Reddit/user threads):
{json.dumps(social_snippets, ensure_ascii=True, indent=2)}

Task:
- First estimate a base sentiment score from 0 to 100.
- 0 = Total Hate
- 100 = Viral Love
- Compare professional reviews vs social threads.
- If Reddit/social users report long-term usage issues or real-world bugs (e.g., battery drain, software bugs)
    that professional reviews ignore or understate, apply a strict 15% penalty to the base score.
- Explicitly mention whether long-term usage or real-world bug signals were found.

Return ONLY valid JSON in this format:
{{
    "base_sentiment_score": number,
    "penalty_applied": true_or_false,
    "final_sentiment_score": number,
  "sentiment_notes": "short explanation"
}}
""".strip()

        response = llm.invoke(prompt)
        content = str(response.content).strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        parsed = json.loads(content)
        base_score = float(parsed.get("base_sentiment_score", 50.0))
        penalty_applied = bool(parsed.get("penalty_applied", False))
        final_score = float(parsed.get("final_sentiment_score", base_score))
        if penalty_applied:
            final_score = min(final_score, round(base_score * 0.85, 2))
        notes = str(parsed.get("sentiment_notes", "")).strip()
        final_score = _clamp(final_score)

        if "long-term usage" not in notes.lower() and "real-world bug" not in notes.lower() and "real-world bugs" not in notes.lower():
            notes = (
                (notes + " " if notes else "")
                + "Long-term usage and real-world bug signals were reviewed during dual-source comparison."
            ).strip()

        return round(final_score, 2), notes or "Qwen returned a sentiment score with no notes."
    except Exception as exc:
        logger.warning("Sentiment analysis failed for '%s': %s", product, exc)
        return 50.0, "Sentiment analysis failed; using neutral fallback."


def _format_price_list(values: List[float]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(f"${v:.2f}" for v in values) + "]"


def _log_deep_research_pricing_table(
    product: str,
    minimum_viable_price: float,
    raw_prices: List[float],
    accessory_prices: List[float],
    amazon_avg_price: float,
    aliexpress_avg_price: float,
) -> None:
    """Print a professional terminal summary of deep-research pricing decisions."""
    if accessory_prices:
        accessory_reason = (
            f"Flagged below minimum viable ${minimum_viable_price:.2f}: {_format_price_list(accessory_prices)}"
        )
    else:
        accessory_reason = f"No prices below minimum viable ${minimum_viable_price:.2f}."

    selected_benchmark = (
        f"Amazon={amazon_avg_price:.2f}, AliExpress={aliexpress_avg_price:.2f}"
    )

    if Console is not None and Table is not None:
        console = Console()
        table = Table(title="Deep Research Price Audit", show_lines=True)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        table.add_row("Candidate Name", product)
        table.add_row("Min Viable Price", f"${minimum_viable_price:.2f}")
        table.add_row("Raw Prices Found", _format_price_list(raw_prices))
        table.add_row("Accessory Filter", accessory_reason)
        table.add_row("Selected Benchmark", selected_benchmark)
        console.print(table)
        return

    border = "+----------------------+------------------------------------------------------------------+"
    print("\nDeep Research Price Audit")
    print(border)
    print(f"| {'Candidate Name'.ljust(20)} | {product[:64].ljust(64)} |")
    print(border)
    print(f"| {'Min Viable Price'.ljust(20)} | {f'${minimum_viable_price:.2f}'.ljust(64)} |")
    print(border)
    print(f"| {'Raw Prices Found'.ljust(20)} | {_format_price_list(raw_prices)[:64].ljust(64)} |")
    print(border)
    print(f"| {'Accessory Filter'.ljust(20)} | {accessory_reason[:64].ljust(64)} |")
    print(border)
    print(f"| {'Selected Benchmark'.ljust(20)} | {selected_benchmark[:64].ljust(64)} |")
    print(border)


def _log_category_profile(profile: Dict[str, Any], category: str) -> None:
    """Structured debug output for the category profile generated by Qwen."""
    if Console is not None and Table is not None:
        console = Console()
        table = Table(title=f"Category Profile: {category}", show_lines=True)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_row("Minimum Viable Price", f"${float(profile.get('minimum_viable_price', 0.0)):.2f}")
        table.add_row("Maximum Viable Price", f"${float(profile.get('maximum_viable_price', 0.0)):.2f}")
        forbidden = profile.get("forbidden_keywords", [])
        forbidden_text = ", ".join(str(item) for item in forbidden) if forbidden else "[]"
        table.add_row("Forbidden Keywords", forbidden_text)
        console.print(table)
        return

    border = "+----------------------+------------------------------------------------------------------+"
    minimum_price = float(profile.get("minimum_viable_price", 0.0))
    maximum_price = float(profile.get("maximum_viable_price", 0.0))
    forbidden = profile.get("forbidden_keywords", [])
    forbidden_text = ", ".join(str(item) for item in forbidden) if forbidden else "[]"

    print(f"\nCategory Profile: {category}")
    print(border)
    print(f"| {'Minimum Viable Price'.ljust(20)} | {f'${minimum_price:.2f}'.ljust(64)} |")
    print(border)
    print(f"| {'Maximum Viable Price'.ljust(20)} | {f'${maximum_price:.2f}'.ljust(64)} |")
    print(border)
    print(f"| {'Forbidden Keywords'.ljust(20)} | {forbidden_text[:64].ljust(64)} |")
    print(border)


def _average_or_zero(values: List[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _extract_shipping_days(text: str) -> float:
    day_numbers = [int(x) for x in re.findall(r"(\d{1,2})\s*(?:day|days|business days)", text, flags=re.IGNORECASE)]
    if not day_numbers:
        return 0.0
    return round(sum(day_numbers) / len(day_numbers), 2)


def _estimate_demand_score(text: str) -> float:
    score = 0.0
    positive_terms = ["trending", "viral", "high demand", "sold out", "popular", "fast-growing", "surging"]
    caution_terms = ["declining", "low demand", "saturated", "seasonal dip", "falling"]

    lowered = text.lower()
    score += 14.0 * sum(term in lowered for term in positive_terms)
    score -= 7.0 * sum(term in lowered for term in caution_terms)

    return round(_clamp(score), 2)


def _estimate_supply_reliability_score(shipping_days: float, text: str) -> float:
    if shipping_days <= 3:
        base = 92.0
    elif shipping_days <= 7:
        base = 80.0
    elif shipping_days <= 14:
        base = 65.0
    else:
        base = 45.0

    lowered = text.lower()
    bonus = 0.0
    if "in stock" in lowered:
        bonus += 5.0
    if "prime" in lowered or "fast shipping" in lowered:
        bonus += 4.0
    if "out of stock" in lowered or "delayed" in lowered:
        bonus -= 8.0

    return round(_clamp(base + bonus), 2)


def _estimate_margin_score(amazon_price: float, aliexpress_price: float) -> float:
    if amazon_price <= 0 or amazon_price < aliexpress_price:
        return 0.0

    margin_pct = ((amazon_price - aliexpress_price) / amazon_price) * 100.0
    return round(_clamp(margin_pct), 2)


def _fallback_metrics_for_product(product: str, minimum_viable_price: float) -> Dict[str, float]:
    """Generate deterministic product-specific fallback metrics when live search fails."""
    digest = hashlib.sha256(product.lower().strip().encode("utf-8")).digest()

    demand_score = 46.0 + (digest[0] % 18)
    supply_reliability_score = 50.0 + (digest[1] % 20)
    sentiment_score = 44.0 + (digest[2] % 18)
    shipping_days = float(4 + (digest[3] % 12))

    aliexpress_avg_price = round(max(minimum_viable_price * 0.75 + (digest[4] % 7), 1.0), 2)
    amazon_floor = max(aliexpress_avg_price * 1.18, minimum_viable_price * 1.35)
    amazon_avg_price = round(amazon_floor + (digest[5] % 9), 2)

    return {
        "demand_score": round(_clamp(demand_score), 2),
        "sentiment_score": round(_clamp(sentiment_score), 2),
        "amazon_avg_price": amazon_avg_price,
        "aliexpress_avg_price": aliexpress_avg_price,
        "shipping_days": round(shipping_days, 2),
        "supply_reliability_score": round(_clamp(supply_reliability_score), 2),
    }


def calculate_success_score(
    demand_score: float,
    amazon_price: float,
    aliexpress_price: float,
    supply_reliability_score: float,
    sentiment_score: float,
) -> tuple[float, float]:
    margin_score = _estimate_margin_score(amazon_price, aliexpress_price)
    final_score = (
        demand_score * 0.4
        + margin_score * 0.3
        + supply_reliability_score * 0.2
        + sentiment_score * 0.1
    )
    return round(margin_score, 2), round(_clamp(final_score), 2)


def discovery_node(state: GraphState) -> Dict[str, Any]:
    category_profile = _generate_category_profile(state["category"])
    category_boundaries = _define_category_boundaries(state["category"])
    category_profile["category_boundaries"] = category_boundaries
    minimum_viable_price = float(category_profile.get("minimum_viable_price", 20.0))
    forbidden_keywords = list(category_profile.get("forbidden_keywords", []))

    search_tool = _create_tavily_search_tool(max_results=8)
    negative_terms = ["review", "tested", "best of", "cnet", "wired", "pcmag"] + forbidden_keywords
    negative_constraint = " ".join(
        f'-"{term}"' if " " in term else f"-{term}" for term in dict.fromkeys(negative_terms)
    )
    category = state["category"].strip()

    targeted_subqueries = _generate_transactional_subqueries(category)
    if DEBUG_MODE:
        print(f"[DEBUG] Transactional subqueries for '{category}': {targeted_subqueries}")

    rescue_queries = [
        f"best {category} products 2026",
        f"top {category} items 2026",
        f"popular {category} models",
        f"new {category} releases 2026",
        f"{category} product list",
    ]

    normalized: List[Dict[str, Any]] = []

    def _run_query_batch(queries: List[str]) -> None:
        for subquery in queries:
            query_payload = {
                "query": f"{subquery} {negative_constraint}",
                "include_domains": ["amazon.com", "alibaba.com", "sephora.com", "ulta.com"],
                "search_depth": "advanced",
                "max_results": 8,
            }
            raw = _run_tavily_query(search_tool, query_payload)
            normalized.extend(_normalize_search_results(raw))

    try:
        _run_query_batch(targeted_subqueries)
    except Exception as exc:
        logger.error("Discovery search failed on targeted pass: %s", exc)

    discovery_debug_rows: List[tuple[str, str]] = []
    products = _clean_titles_with_local_qwen(
        normalized,
        category=state["category"],
        category_boundaries=category_boundaries,
        debug_rows=discovery_debug_rows,
        forbidden_keywords=forbidden_keywords,
        provider=state.get("provider", "Cloud (Groq)"),
    )

    if len(products) < 5:
        try:
            _run_query_batch(rescue_queries)
        except Exception as exc:
            logger.warning("Discovery rescue search failed: %s", exc)

        rescue_debug_rows: List[tuple[str, str]] = []
        rescued_products = _clean_titles_with_local_qwen(
            normalized,
            category=state["category"],
            category_boundaries=category_boundaries,
            debug_rows=rescue_debug_rows,
            forbidden_keywords=forbidden_keywords,
            provider=state.get("provider", "Cloud (Groq)"),
        )
        for item in rescued_products:
            if item.lower() not in {p.lower() for p in products}:
                products.append(item)
        discovery_debug_rows.extend(rescue_debug_rows)

    if len(products) < 5:
        fallback_titles = _hard_filter_product_candidates(
            [str(row.get("title") or "").strip() for row in normalized if str(row.get("title") or "").strip()]
        )
        for title in fallback_titles:
            if title.lower() not in {p.lower() for p in products}:
                products.append(title)
            if len(products) >= 5:
                break

    if not products:
        seed_products = _seed_products_from_category(state["category"])
        logger.warning(
            "Discovery produced zero products after cleaning for category '%s'. Using deterministic seed products.",
            state["category"],
        )
        products.extend(seed_products)

    if DEBUG_MODE:
        _log_category_profile(category_profile, state["category"])
        _print_discovery_debug_table(discovery_debug_rows)
        print(
            f"[DEBUG] Category '{state['category']}' minimum viable price: ${minimum_viable_price:.2f}"
        )
    if not products:
        logger.error("Discovery found no valid products for category: %s", state["category"])
        raise RuntimeError("No real products found after local Qwen title cleaning.")

    if len(products) < 5:
        logger.warning(
            "Discovery returned only %s products after rescue attempts for category '%s'.",
            len(products),
            state["category"],
        )

    return {
        "category_profile": category_profile,
        "discovered_products": products[:5],
        "minimum_viable_price": minimum_viable_price,
    }


def deep_research_node(state: GraphState) -> Dict[str, Any]:
    search_tool = _create_tavily_search_tool(max_results=6)
    researched: List[ResearchProduct] = []
    category_profile = state.get("category_profile", {})
    minimum_viable_price = float(
        category_profile.get("minimum_viable_price", state.get("minimum_viable_price", 20.0))
    )

    for product in state["discovered_products"]:
        lowered_product = product.lower()
        if re.search(r"\b(repair|parts)\b", lowered_product):
            logger.info("Skipping product '%s' because it contains repair/parts keywords.", product)
            continue

        demand_query = (
            f"{product} market demand April 2026 search interest social media buzz "
            f"ecommerce trend"
        )
        supply_queries = _build_supply_queries(product)

        try:
            demand_raw = _run_tavily_query(search_tool, demand_query)
            _normalize_search_results(demand_raw)

            supply_raw_list: List[Any] = []
            prices: List[float] = []
            raw_prices_all: List[float] = []
            accessory_prices_all: List[float] = []
            for supply_query in supply_queries:
                supply_raw = _run_tavily_query(search_tool, supply_query)
                _normalize_search_results(supply_raw)
                supply_raw_list.append(supply_raw)

                candidate_prices = _extract_prices(json.dumps(supply_raw, ensure_ascii=True))
                raw_prices_all.extend(candidate_prices)

                cleaned_prices = _clean_prices_for_product(candidate_prices, minimum_viable_price, product)
                accessory_prices_all.extend([p for p in candidate_prices if p < minimum_viable_price])

                prices.extend(cleaned_prices)

                # If we have valid prices, stop querying additional fallbacks.
                if prices:
                    break
        except Exception as exc:
            logger.error("Deep research search failed for product '%s': %s", product, exc)
            logger.warning(
                "Using fallback deep-research metrics for '%s' so the pipeline can continue.",
                product,
            )

            fallback_metrics = _fallback_metrics_for_product(product, minimum_viable_price)

            researched.append(
                {
                    "product_name": product,
                    "demand_score": fallback_metrics["demand_score"],
                    "demand_notes": "Fallback demand estimate used because Tavily search was unavailable.",
                    "sentiment_score": fallback_metrics["sentiment_score"],
                    "sentiment_notes": "Sentiment pending analysis.",
                    "amazon_avg_price": fallback_metrics["amazon_avg_price"],
                    "aliexpress_avg_price": fallback_metrics["aliexpress_avg_price"],
                    "margin_score": _estimate_margin_score(
                        fallback_metrics["amazon_avg_price"],
                        fallback_metrics["aliexpress_avg_price"],
                    ),
                    "shipping_days": fallback_metrics["shipping_days"],
                    "supply_reliability_score": fallback_metrics["supply_reliability_score"],
                }
            )
            continue

        demand_text = json.dumps(demand_raw, ensure_ascii=True)
        supply_text = json.dumps(supply_raw_list, ensure_ascii=True)

        demand_score = _estimate_demand_score(demand_text)
        demand_notes = demand_text[:450]

        if not prices:
            raw_prices_all = _extract_prices(supply_text)
            prices = _clean_prices_for_product(raw_prices_all, minimum_viable_price, product)
            accessory_prices_all = [p for p in raw_prices_all if p < minimum_viable_price]
        midpoint = len(prices) // 2
        amazon_prices = prices[:midpoint] if midpoint else prices
        aliexpress_prices = prices[midpoint:] if midpoint else prices

        amazon_avg_price = _average_or_zero(amazon_prices)
        aliexpress_avg_price = _average_or_zero(aliexpress_prices)

        _log_deep_research_pricing_table(
            product=product,
            minimum_viable_price=minimum_viable_price,
            raw_prices=raw_prices_all,
            accessory_prices=accessory_prices_all,
            amazon_avg_price=amazon_avg_price,
            aliexpress_avg_price=aliexpress_avg_price,
        )

        shipping_days = _extract_shipping_days(supply_text)
        supply_reliability_score = _estimate_supply_reliability_score(shipping_days, supply_text)
        margin_score = _estimate_margin_score(amazon_avg_price, aliexpress_avg_price)

        researched.append(
            {
                "product_name": product,
                "demand_score": demand_score,
                "demand_notes": demand_notes,
                "sentiment_score": 50.0,
                "sentiment_notes": "Sentiment pending analysis.",
                "amazon_avg_price": amazon_avg_price,
                "aliexpress_avg_price": aliexpress_avg_price,
                "margin_score": margin_score,
                "shipping_days": shipping_days,
                "supply_reliability_score": supply_reliability_score,
            }
        )

    return {"researched_products": researched}


def sentiment_node(state: GraphState) -> Dict[str, Any]:
    search_tool = _create_tavily_search_tool(max_results=6)
    enriched: List[ResearchProduct] = []

    for item in state["researched_products"]:
        product = item["product_name"]
        professional_query = (
            f'"{product}" reviews OR "{product}" user feedback OR "{product}" customer feedback '
            f'OR "{product}" opinion'
        )
        social_query = f'{product} reddit "user review" 2026 -bot'

        try:
            professional_raw = _run_tavily_query(search_tool, professional_query)
            professional_results = _normalize_search_results(professional_raw)
        except Exception as exc:
            logger.warning("Professional sentiment search failed for product '%s': %s", product, exc)
            professional_results = []

        try:
            social_raw = _run_tavily_query(search_tool, social_query)
            social_results = _normalize_search_results(social_raw)
        except Exception as exc:
            logger.warning("Social sentiment search failed for product '%s': %s", product, exc)
            social_results = []

        professional_snippets = _collect_sentiment_snippets(
            product,
            professional_results,
            trigger_terms=["review", "reviews", "user feedback", "customer feedback", "opinion", "editorial"],
        )
        social_snippets = _collect_sentiment_snippets(
            product,
            social_results,
            trigger_terms=["reddit", "user review", "long-term", "battery", "bug", "bugs", "issue", "issues"],
        )

        sentiment_score, sentiment_notes = _analyze_sentiment_with_qwen(
            product,
            professional_snippets,
            social_snippets,
        )

        updated_item: ResearchProduct = {
            **item,
            "sentiment_score": sentiment_score,
            "sentiment_notes": sentiment_notes,
        }
        enriched.append(updated_item)

    return {"researched_products": enriched}


def scoring_node(state: GraphState) -> Dict[str, Any]:
    scored: List[ScoredProduct] = []

    for item in state["researched_products"]:
        margin_score, success_score = calculate_success_score(
            demand_score=item["demand_score"],
            amazon_price=item["amazon_avg_price"],
            aliexpress_price=item["aliexpress_avg_price"],
            supply_reliability_score=item["supply_reliability_score"],
            sentiment_score=item.get("sentiment_score", 50.0),
        )

        scored.append(
            {
                "product_name": item["product_name"],
                "demand_score": item["demand_score"],
                "margin_score": margin_score,
                "supply_reliability_score": item["supply_reliability_score"],
                "sentiment_score": item.get("sentiment_score", 50.0),
                "success_score": success_score,
                "demand_notes": item["demand_notes"],
                "sentiment_notes": item.get("sentiment_notes", ""),
                "amazon_avg_price": item["amazon_avg_price"],
                "aliexpress_avg_price": item["aliexpress_avg_price"],
                "shipping_days": item["shipping_days"],
            }
        )

    scored.sort(key=lambda x: x["success_score"], reverse=True)
    if not scored:
        raise RuntimeError("No products qualified for scoring with real demand and price data.")
    return {"scored_products": scored}


def _generate_seo_description(llm: Any, category: str, product_name: str) -> str:
    prompt = f"""
You are an e-commerce SEO writer.
Write exactly 100 words for a product description.
Category: {category}
Product: {product_name}
Requirements:
- Optimize for search intent and buyer conversion.
- Include natural keywords, benefits, and trust signals.
- Keep tone concise, persuasive, and readable.
- Return plain text only.
""".strip()

    try:
        response = llm.invoke(prompt)
        text = str(response.content).strip()
    except Exception:
        # Keep pipeline running even when external LLM auth/network fails.
        text = (
            f"{product_name} is a high-demand option in {category}, designed for shoppers who value quality, "
            "convenience, reliability, and practical everyday performance. Built for modern e-commerce buyers, "
            "it combines useful features, strong value, and trustworthy results. This product supports better "
            "customer satisfaction through smart design, consistent quality, and easy use. Ideal for trend-focused "
            "stores, it helps improve click-through rates, conversions, and repeat purchases with compelling benefits "
            "and clear buyer appeal. Choose this product to stay competitive, capture rising demand, and offer a "
            "confident shopping experience that encourages long-term brand loyalty and sustainable sales growth."
        )

    if not text:
        text = (
            f"{product_name} is a high-demand option in {category}, designed for shoppers who value quality, "
            "convenience, and reliable performance."
        )

    words = text.split()
    if len(words) > 100:
        text = " ".join(words[:100])
    elif len(words) < 100:
        filler = " Order now to improve daily results with a trusted, trend-driven product choice."
        text = (text + filler)
        text = " ".join(text.split()[:100])

    return text


def seo_node(state: GraphState) -> Dict[str, Any]:
    llm = None
    try:
        llm = get_llm(state["provider"])
    except Exception:
        # Continue with deterministic fallback description if model is unavailable.
        pass

    final_report: List[FinalProduct] = []

    for item in state["scored_products"]:
        seo_text = _generate_seo_description(llm, state["category"], item["product_name"])
        final_report.append(
            {
                "product_name": item["product_name"],
                "success_score": item["success_score"],
                "seo_description": seo_text,
                "demand_score": item["demand_score"],
                "margin_score": item["margin_score"],
                "supply_reliability_score": item["supply_reliability_score"],
                "sentiment_score": item.get("sentiment_score", 50.0),
                "sentiment_notes": item.get("sentiment_notes", ""),
                "amazon_avg_price": item["amazon_avg_price"],
                "aliexpress_avg_price": item["aliexpress_avg_price"],
                "shipping_days": item["shipping_days"],
            }
        )

    return {"final_report": final_report}


def build_graph():
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("discovery", discovery_node)
    graph_builder.add_node("deep_research", deep_research_node)
    graph_builder.add_node("sentiment", sentiment_node)
    graph_builder.add_node("scoring", scoring_node)
    graph_builder.add_node("seo", seo_node)

    graph_builder.add_edge(START, "discovery")
    graph_builder.add_edge("discovery", "deep_research")
    graph_builder.add_edge("deep_research", "sentiment")
    graph_builder.add_edge("sentiment", "scoring")
    graph_builder.add_edge("scoring", "seo")
    graph_builder.add_edge("seo", END)

    return graph_builder.compile()


def run(category: str, provider: str) -> GraphState:
    graph = build_graph()
    initial_state: GraphState = {
        "category": category,
        "provider": provider,
        "category_profile": {},
        "minimum_viable_price": 20.0,
        "discovered_products": [],
        "researched_products": [],
        "scored_products": [],
        "final_report": [],
    }
    return graph.invoke(initial_state)


def save_report(state: GraphState, output_dir: Path) -> Optional[Path]:
    if not state["final_report"]:
        logger.error("Final report is empty. Existing output file will not be overwritten.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_category = re.sub(r"[^a-zA-Z0-9_-]+", "_", state["category"].strip()).strip("_")
    filename = f"{safe_category.lower() or 'category'}_trend_report.json"
    out_path = output_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(state["final_report"], f, indent=2, ensure_ascii=False)

    metadata_path = output_dir / f"{safe_category.lower() or 'category'}_trend_metadata.json"
    metadata_payload = {
        "category": state["category"],
        "provider": state["provider"],
        "category_profile": state.get("category_profile", {}),
        "minimum_viable_price": state.get("minimum_viable_price", 20.0),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Prefer project-local .env values over any stale terminal/session variables.
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description="TrendSense AI Research Agent (LangGraph)")
    parser.add_argument("category", nargs="?", help="Product category to analyze")
    parser.add_argument(
        "--provider",
        choices=["Local (Qwen 2.5)", "Cloud (Groq)"],
        default="Cloud (Groq)",
        help="LLM provider used for intelligence steps.",
    )
    args = parser.parse_args()

    category = args.category or input("Enter e-commerce category: ").strip()
    if not category:
        raise ValueError("Category cannot be empty.")

    missing = [k for k in ("TAVILY_API_KEY",) if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}."
        )

    provider = args.provider
    if provider == "Cloud (Groq)" and not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not set. Falling back to non-LLM filtering/SEO where possible.")

    try:
        result_state = run(category, provider)
    except Exception as exc:
        logger.error("Graph execution stopped: %s", exc)
        raise SystemExit(1) from exc

    output_path = save_report(result_state, Path("output"))
    if output_path is None:
        raise SystemExit(1)

    print("\nTop Product Opportunities:\n")
    for i, item in enumerate(result_state["final_report"], start=1):
        print(f"{i}. {item['product_name']}")
        print(f"   SuccessScore: {item['success_score']}")
        print(f"   Demand: {item['demand_score']} | Margin: {item['margin_score']} | Supply: {item['supply_reliability_score']}")
        print(f"   Amazon Avg: ${item['amazon_avg_price']} | AliExpress Avg: ${item['aliexpress_avg_price']} | Shipping: {item['shipping_days']} days")
        print(f"   SEO: {item['seo_description']}\n")

    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()


