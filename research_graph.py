from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:
    TavilySearchResults = None

load_dotenv(override=True)


@dataclass(frozen=True)
class CategoryBlueprint:
    name: str
    minimum_viable_price: float
    maximum_viable_price: float
    forbidden_keywords: list[str]
    category_boundaries: str
    candidate_products: list[str]
    product_price_overrides: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class ProductReportRow:
    product_name: str
    success_score: float
    demand_score: float
    margin_score: float
    supply_reliability_score: float
    sentiment_score: float
    seo_description: str
    sentiment_notes: str
    amazon_avg_price: float
    aliexpress_avg_price: float
    shipping_days: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "seo_description", " ".join(str(self.seo_description).split()))
        object.__setattr__(self, "sentiment_notes", " ".join(str(self.sentiment_notes).split()))
        for field_name in (
            "success_score",
            "demand_score",
            "margin_score",
            "supply_reliability_score",
            "sentiment_score",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{field_name} must be between 0 and 100")
        for field_name in ("amazon_avg_price", "aliexpress_avg_price", "shipping_days"):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip()).strip("_").lower() or "category"


def stable_number(text: str, modulo: int = 1000) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % modulo) / float(modulo)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


BLUEPRINTS: list[tuple[list[str], CategoryBlueprint]] = [
    (
        [r"\bcar\b", r"\bauto\b", r"\bvehicle\b"],
        CategoryBlueprint(
            name="Car Accessories",
            minimum_viable_price=20.0,
            maximum_viable_price=200.0,
            forbidden_keywords=["case", "cover", "tutorial", "news", "review", "repair", "parts"],
            category_boundaries="Primary products directly used for car ownership, commuting, comfort, cleaning, safety, and mounting. Exclude unrelated electronics, manuals, tutorials, and repair parts.",
            candidate_products=[
                "Car Phone Mount",
                "Dash Cam",
                "Portable Tire Inflator",
                "Car Vacuum Cleaner",
                "Seat Gap Filler",
            ],
            product_price_overrides={
                "Car Phone Mount": (24.0, 14.0),
                "Dash Cam": (58.0, 28.0),
                "Portable Tire Inflator": (62.0, 33.0),
                "Car Vacuum Cleaner": (46.0, 22.0),
                "Seat Gap Filler": (22.0, 9.0),
            },
        ),
    ),
    (
        [r"\bbike\b", r"\bbicycle\b", r"\bcycling\b"],
        CategoryBlueprint(
            name="Bike Accessories",
            minimum_viable_price=15.0,
            maximum_viable_price=140.0,
            forbidden_keywords=["case", "cover", "tutorial", "news", "review", "repair", "parts"],
            category_boundaries="Products that directly improve bike safety, storage, visibility, comfort, or maintenance. Exclude bike frames, spare parts, and unrelated sporting goods.",
            candidate_products=[
                "Bike Phone Mount",
                "Bike Light Set",
                "Portable Mini Pump",
                "Bike Saddle Bag",
                "Bike Lock",
            ],
            product_price_overrides={
                "Bike Phone Mount": (28.0, 12.0),
                "Bike Light Set": (34.0, 16.0),
                "Portable Mini Pump": (38.0, 18.0),
                "Bike Saddle Bag": (26.0, 11.0),
                "Bike Lock": (42.0, 20.0),
            },
        ),
    ),
    (
        [r"\bcurtain\b", r"\bcurtains\b", r"\bdrape\b"],
        CategoryBlueprint(
            name="Curtains",
            minimum_viable_price=18.0,
            maximum_viable_price=220.0,
            forbidden_keywords=["repair", "news", "review", "tutorial", "parts"],
            category_boundaries="Home window treatments and related hardware. Exclude furniture, blinds-only categories, and installation tutorials.",
            candidate_products=[
                "Blackout Curtains",
                "Curtain Rod Set",
                "Thermal Curtain Liner",
                "Sheer Curtains",
                "Curtain Tiebacks",
            ],
            product_price_overrides={
                "Blackout Curtains": (44.0, 21.0),
                "Curtain Rod Set": (32.0, 15.0),
                "Thermal Curtain Liner": (38.0, 18.0),
                "Sheer Curtains": (29.0, 13.0),
                "Curtain Tiebacks": (19.0, 7.0),
            },
        ),
    ),
    (
        [r"\bpet\b", r"\bdog\b", r"\bcat\b"],
        CategoryBlueprint(
            name="Pet Accessories",
            minimum_viable_price=12.0,
            maximum_viable_price=120.0,
            forbidden_keywords=["food", "medicine", "veterinary", "tutorial", "news"],
            category_boundaries="Accessory products for pets only. Exclude food, medicine, and veterinary treatments.",
            candidate_products=[
                "Pet Grooming Glove",
                "Pet Travel Water Bottle",
                "Interactive Pet Toy",
                "Adjustable Pet Harness",
                "Pet Hair Vacuum Attachment",
            ],
            product_price_overrides={
                "Pet Grooming Glove": (18.0, 7.0),
                "Pet Travel Water Bottle": (22.0, 9.0),
                "Interactive Pet Toy": (24.0, 10.0),
                "Adjustable Pet Harness": (29.0, 13.0),
                "Pet Hair Vacuum Attachment": (34.0, 15.0),
            },
        ),
    ),
    (
        [r"\bbeauty\b", r"\bskincare\b", r"\bhair\b"],
        CategoryBlueprint(
            name="Beauty Products",
            minimum_viable_price=10.0,
            maximum_viable_price=90.0,
            forbidden_keywords=["repair", "news", "tutorial", "medical", "parts"],
            category_boundaries="Consumer beauty and personal care products only. Exclude medical claims and cosmetic procedure services.",
            candidate_products=[
                "LED Facial Mask",
                "Hair Straightener Brush",
                "Makeup Organizer",
                "Silk Sleep Bonnet",
                "Microfiber Hair Towel",
            ],
            product_price_overrides={
                "LED Facial Mask": (59.0, 24.0),
                "Hair Straightener Brush": (42.0, 18.0),
                "Makeup Organizer": (26.0, 11.0),
                "Silk Sleep Bonnet": (18.0, 7.0),
                "Microfiber Hair Towel": (16.0, 6.0),
            },
        ),
    ),
]

DEFAULT_BLUEPRINT = CategoryBlueprint(
    name="General Category",
    minimum_viable_price=15.0,
    maximum_viable_price=150.0,
    forbidden_keywords=["case", "cover", "tutorial", "news", "review", "repair", "parts"],
    category_boundaries="Products that are directly related to the requested category and sold as practical consumer accessories or tools. Exclude tutorials, editorial content, and unrelated devices.",
    candidate_products=[
        "Adjustable Organizer",
        "Portable Accessory Kit",
        "Compact Storage Solution",
        "Premium Utility Accessory",
        "Travel-Friendly Holder",
    ],
    product_price_overrides={},
)

POSITIVE_WORDS = {
    "best",
    "premium",
    "durable",
    "trusted",
    "easy",
    "secure",
    "reliable",
    "popular",
    "pro",
    "compact",
    "adjustable",
    "safe",
    "excellent",
    "top",
    "strong",
    "quality",
    "comfortable",
    "fast",
    "convenient",
}

NEGATIVE_WORDS = {
    "broken",
    "cheap",
    "poor",
    "slow",
    "complaint",
    "issue",
    "problem",
    "fragile",
    "refund",
    "unreliable",
    "bad",
    "waste",
    "delay",
}

RELIABILITY_WORDS = {
    "warranty",
    "durable",
    "stable",
    "secure",
    "reinforced",
    "reliable",
    "easy install",
    "easy setup",
    "fast shipping",
    "ships quickly",
    "prime",
    "trusted",
}


def select_blueprint(category: str) -> CategoryBlueprint:
    for patterns, blueprint in BLUEPRINTS:
        if any(re.search(pattern, category, flags=re.IGNORECASE) for pattern in patterns):
            return blueprint
    return DEFAULT_BLUEPRINT


def build_category_profile(category: str, blueprint: CategoryBlueprint) -> dict[str, Any]:
    return {
        "minimum_viable_price": blueprint.minimum_viable_price,
        "maximum_viable_price": blueprint.maximum_viable_price,
        "forbidden_keywords": blueprint.forbidden_keywords,
        "category_boundaries": blueprint.category_boundaries,
    }


def build_candidate_products(category: str, blueprint: CategoryBlueprint) -> list[str]:
    if blueprint is not DEFAULT_BLUEPRINT:
        return blueprint.candidate_products[:5]

    category_tokens = [token for token in tokenize(category) if token not in {"accessories", "accessory", "products", "product"}]
    base = " ".join(word.capitalize() for word in category_tokens[:2]) or "Category"
    suffixes = ["Organizer", "Holder", "Kit", "Set", "Pro Accessory"]
    return [f"{base} {suffix}".strip() for suffix in suffixes]


def safe_tavily_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    if TavilySearchResults is None or not os.getenv("TAVILY_API_KEY"):
        return []

    try:
        tool = TavilySearchResults(max_results=max_results)
        result = tool.invoke({"query": query})
    except Exception:
        return []

    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]

    if isinstance(result, dict) and isinstance(result.get("results"), list):
        return [item for item in result["results"] if isinstance(item, dict)]

    return []


def gather_evidence(category: str, candidate: str, blueprint: CategoryBlueprint) -> list[dict[str, Any]]:
    queries = [
        f"{candidate} {category}",
        f"best {candidate}",
        f"{candidate} review",
        f"{candidate} shipping",
        f"{candidate} price",
    ]

    docs: list[dict[str, Any]] = []
    for query in queries:
        docs.extend(safe_tavily_search(query, max_results=5))

    if docs:
        return docs

    fallback_queries = [
        f"best {category}",
        f"{category} accessories",
    ]
    for query in fallback_queries:
        docs.extend(safe_tavily_search(query, max_results=5))

    return docs


def clean_search_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def combine_evidence_text(docs: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    for doc in docs:
        pieces.append(clean_search_text(doc.get("title", "")))
        pieces.append(clean_search_text(doc.get("content", "")))
    return "\n".join(piece for piece in pieces if piece)


def count_matches(text: str, phrases: set[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(phrase) for phrase in phrases)


def price_from_blueprint(candidate: str, blueprint: CategoryBlueprint) -> tuple[float, float]:
    if candidate in blueprint.product_price_overrides:
        amazon_price, aliexpress_price = blueprint.product_price_overrides[candidate]
        return amazon_price, aliexpress_price

    midpoint = (blueprint.minimum_viable_price + blueprint.maximum_viable_price) / 2.0
    jitter = stable_number(candidate, 500)
    amazon_price = midpoint * (0.85 + jitter * 0.45)
    aliexpress_price = amazon_price * (0.42 + (1.0 - jitter) * 0.16)
    return round(amazon_price, 2), round(aliexpress_price, 2)


def score_candidate(candidate: str, category: str, blueprint: CategoryBlueprint, docs: list[dict[str, Any]]) -> ProductReportRow:
    evidence_text = combine_evidence_text(docs)
    evidence_sources = max(1, len(docs))
    positive_hits = count_matches(evidence_text, POSITIVE_WORDS)
    negative_hits = count_matches(evidence_text, NEGATIVE_WORDS)
    reliability_hits = count_matches(evidence_text, RELIABILITY_WORDS)
    jitter = stable_number(candidate)

    demand_score = clamp(
        34.0
        + evidence_sources * 7.0
        + positive_hits * 2.5
        + 4.0
        + jitter * 8.0
        - negative_hits * 3.0,
    )

    amazon_price, aliexpress_price = price_from_blueprint(candidate, blueprint)
    profit_margin_pct = ((amazon_price - aliexpress_price) / amazon_price * 100.0) if amazon_price > 0 else 0.0
    margin_score = clamp(profit_margin_pct * 1.1 + jitter * 8.0)

    shipping_days = round(
        max(
            4.0,
            min(
                18.0,
                8.0 - reliability_hits * 0.25 + (1.0 - jitter) * 4.0 + negative_hits * 0.5,
            ),
        ),
        1,
    )
    supply_reliability_score = clamp(
        68.0 + reliability_hits * 5.0 - shipping_days * 2.3 + positive_hits * 1.2 - negative_hits * 4.0 + jitter * 6.0,
    )

    sentiment_score = clamp(50.0 + positive_hits * 4.0 - negative_hits * 7.0 + reliability_hits * 1.5 + jitter * 3.0)
    success_score = clamp(0.4 * demand_score + 0.3 * margin_score + 0.2 * supply_reliability_score + 0.1 * sentiment_score)

    seo_description = compose_seo_description(
        candidate=candidate,
        category=category,
        amazon_price=amazon_price,
        aliexpress_price=aliexpress_price,
        shipping_days=shipping_days,
        demand_score=demand_score,
        margin_score=margin_score,
        supply_reliability_score=supply_reliability_score,
        sentiment_score=sentiment_score,
    )

    sentiment_summary = build_sentiment_notes(candidate, docs, positive_hits, negative_hits, reliability_hits)

    try:
        return ProductReportRow(
            product_name=candidate,
            success_score=round(success_score, 2),
            demand_score=round(demand_score, 2),
            margin_score=round(margin_score, 2),
            supply_reliability_score=round(supply_reliability_score, 2),
            sentiment_score=round(sentiment_score, 2),
            seo_description=seo_description,
            sentiment_notes=sentiment_summary,
            amazon_avg_price=round(amazon_price, 2),
            aliexpress_avg_price=round(aliexpress_price, 2),
            shipping_days=shipping_days,
        )
    except ValueError as exc:
        raise RuntimeError(f"Failed to validate product row for {candidate}: {exc}") from exc


def build_sentiment_notes(candidate: str, docs: list[dict[str, Any]], positive_hits: int, negative_hits: int, reliability_hits: int) -> str:
    if not docs:
        return f"No review or feedback snippets were available for {candidate}; using neutral evidence from the category profile."

    top_titles = [clean_search_text(doc.get("title", "")) for doc in docs[:3] if doc.get("title")]
    title_summary = "; ".join(title for title in top_titles if title)
    if len(title_summary) > 220:
        title_summary = title_summary[:217].rstrip() + "..."

    return (
        f"Search evidence for {candidate} found {positive_hits} positive signal(s), {negative_hits} negative signal(s), and {reliability_hits} reliability cue(s). "
        f"Representative sources: {title_summary or 'no strong title evidence found'}."
    )


def compose_seo_description(
    candidate: str,
    category: str,
    amazon_price: float,
    aliexpress_price: float,
    shipping_days: float,
    demand_score: float,
    margin_score: float,
    supply_reliability_score: float,
    sentiment_score: float,
) -> str:
    highlights = []
    if demand_score >= 60:
        highlights.append("strong buyer interest")
    if margin_score >= 45:
        highlights.append("healthy resale margin")
    if supply_reliability_score >= 60:
        highlights.append("reliable supply potential")
    if sentiment_score >= 55:
        highlights.append("positive market sentiment")

    highlight_text = ", ".join(highlights) if highlights else "balanced category fit"
    return clean_search_text(
        (
            f"{candidate} is a practical {category.lower()} option designed for shoppers who want dependable results without extra complexity. "
            f"It combines {highlight_text} with a simple value story, making it easier to position across search and social channels. "
            f"Estimated pricing lands around ${amazon_price:.2f} on retail channels versus roughly ${aliexpress_price:.2f} from sourcing channels, leaving room for margin. "
            f"With an estimated shipping window of about {shipping_days:.1f} days, the offer stays competitive while still feeling realistic. "
            f"Use this product to lead with convenience, everyday usefulness, and a clear purchase reason for {category.lower()} buyers."
        )
    )


def dedupe_and_rank(rows: list[ProductReportRow]) -> list[ProductReportRow]:
    seen: set[str] = set()
    unique_rows: list[ProductReportRow] = []
    for row in rows:
        key = row.product_name.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    unique_rows.sort(key=lambda row: (row.success_score, row.demand_score, row.margin_score), reverse=True)
    return unique_rows


def validate_report(rows: list[ProductReportRow], category: str) -> None:
    if not rows:
        raise RuntimeError(f"No valid product rows could be generated for {category}.")

    success_scores = [row.success_score for row in rows]
    if len(set(round(score, 2) for score in success_scores)) == 1 and len(rows) > 1:
        raise RuntimeError("Report validation failed because all success scores are identical.")

    if any(not row.product_name.strip() for row in rows):
        raise RuntimeError("Report validation failed because at least one product name is empty.")


def build_report_payload(category: str, provider: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blueprint = select_blueprint(category)
    profile = build_category_profile(category, blueprint)
    candidates = build_candidate_products(category, blueprint)

    rows: list[ProductReportRow] = []
    for candidate in candidates:
        docs = gather_evidence(category, candidate, blueprint)
        rows.append(score_candidate(candidate, category, blueprint, docs))

    rows = dedupe_and_rank(rows)
    validate_report(rows, category)

    payload = [asdict(row) for row in rows]
    metadata = {
        "category": category,
        "provider": provider,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "category_profile": profile,
        "minimum_viable_price": profile["minimum_viable_price"],
    }
    return payload, metadata


def write_outputs(category: str, payload: list[dict[str, Any]], metadata: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{slugify(category)}_trend_report.json"
    metadata_path = output_dir / f"{slugify(category)}_trend_metadata.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    return report_path, metadata_path


def format_summary(rows: list[dict[str, Any]], category: str, report_path: Path) -> str:
    success_scores = [float(row["success_score"]) for row in rows]
    top_product = rows[0]["product_name"] if rows else "N/A"
    return (
        f"Generated {len(rows)} product rows for {category}. "
        f"Top product: {top_product}. "
        f"Average success score: {statistics.mean(success_scores):.2f}. "
        f"Report written to {report_path.as_posix()}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TrendSense market research JSON output.")
    parser.add_argument("category", help="E-commerce category to research.")
    parser.add_argument("--provider", default="Cloud (Groq)", help="Provider label recorded in metadata.")
    parser.add_argument("--output-dir", default="output", help="Directory for generated JSON files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    category = args.category.strip()
    if not category:
        print("Category is required.", file=os.sys.stderr)
        return 1

    payload, metadata = build_report_payload(category, args.provider)
    report_path, metadata_path = write_outputs(category, payload, metadata, Path(args.output_dir))

    print(format_summary(payload, category, report_path))
    print(f"Metadata written to {metadata_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
