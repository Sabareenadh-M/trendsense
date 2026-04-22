from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any
import hashlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

# Wide layout improves readability for KPI rows and horizontal charts.
st.set_page_config(page_title="TrendSense Dashboard", layout="wide")

# Minimal CSS to keep a clean, professional UI tone.
st.markdown(
    """
    <style>
    .seo-box {
        border: 1px solid #d9d9d9;
        border-radius: 8px;
        padding: 0.75rem;
        background: #fafafa;
    }
    .small-note {
        color: #666666;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULT_REPORT_PATH = Path("output/beauty_products_trend_report.json")
RESEARCH_SCRIPT_PATH = Path("research_graph.py")
PROVIDER_OPTIONS = ["Local (Qwen 2.5)", "Cloud (Groq)"]
USERS_FILE_PATH = Path("data/users.json")

# Ensure API keys from project .env are available inside Streamlit runtime.
load_dotenv(override=True)


def initialize_session_state() -> None:
    """Create stable keys so data persists across tabs and reruns."""
    if "research_data" not in st.session_state:
        st.session_state.research_data = []
    if "research_results" not in st.session_state:
        st.session_state.research_results = []
    if "category_profile" not in st.session_state:
        st.session_state.category_profile = {}
    if "research_category" not in st.session_state:
        st.session_state.research_category = ""
    if "research_file_path" not in st.session_state:
        st.session_state.research_file_path = str(DEFAULT_REPORT_PATH)
    if "ai_outputs" not in st.session_state:
        st.session_state.ai_outputs = {}
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = PROVIDER_OPTIONS[0]
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = ""


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Create a PBKDF2 hash and return (salt_hex, hash_hex)."""
    use_salt = salt or secrets.token_bytes(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), use_salt, 100_000)
    return use_salt.hex(), pwd_hash.hex()


def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    """Validate plain password against stored PBKDF2 hash."""
    try:
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)
    except ValueError:
        return False

    check_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return secrets.compare_digest(check_hash, expected_hash)


def _load_users() -> dict[str, dict[str, str]]:
    """Load saved users from data/users.json."""
    if not USERS_FILE_PATH.exists():
        return {}

    try:
        with USERS_FILE_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_users(users: dict[str, dict[str, str]]) -> None:
    """Persist user records in data/users.json."""
    USERS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USERS_FILE_PATH.open("w", encoding="utf-8") as file:
        json.dump(users, file, indent=2)


def _validate_signup_password(password: str) -> str | None:
    """Return validation message for weak password, otherwise None."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return "Password must include at least one uppercase letter."
    if not re.search(r"[^A-Za-z0-9]", password):
        return "Password must include at least one special character."
    return None


def _render_auth_theme() -> None:
    """Inject auth-page styling for a polished login/signup experience."""
    st.markdown(
        """
        <style>
        :root {
            --auth-bg-a: #eef9f4;
            --auth-bg-b: #e6f2ff;
            --auth-accent: #0f766e;
            --auth-accent-2: #0284c7;
            --auth-text: #102a43;
            --auth-muted: #486581;
            --auth-border: #d9e2ec;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 8% 12%, rgba(15, 118, 110, 0.12), transparent 34%),
                radial-gradient(circle at 88% 15%, rgba(2, 132, 199, 0.12), transparent 32%),
                linear-gradient(135deg, var(--auth-bg-a), var(--auth-bg-b));
        }

        .auth-title {
            color: var(--auth-text);
            font-weight: 700;
            margin-bottom: 0.25rem;
            letter-spacing: 0.2px;
        }

        .auth-subtitle {
            color: var(--auth-muted);
            margin-bottom: 0.8rem;
        }

        .auth-shell {
            background: rgba(255, 255, 255, 0.80);
            border: 1px solid var(--auth-border);
            border-radius: 16px;
            padding: 0.8rem;
            box-shadow: 0 14px 34px rgba(16, 42, 67, 0.08);
        }

        [data-testid="stTabs"] button {
            border-radius: 10px;
            font-weight: 600;
        }

        [data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #e4ecf3;
            border-radius: 12px;
            padding: 0.75rem;
        }

        [data-testid="stTextInput"] label {
            color: var(--auth-text);
            font-weight: 600;
        }

        [data-testid="stTextInput"] input {
            border: 1px solid var(--auth-border);
            border-radius: 10px;
        }

        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            background: linear-gradient(90deg, var(--auth-accent), var(--auth-accent-2));
            border: none;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_auth_pages() -> bool:
    """Render login/signup pages and return auth status."""
    _render_auth_theme()
    left_col, center_col, right_col = st.columns([1, 1.35, 1])
    with center_col:
        st.markdown('<h1 class="auth-title">Welcome to TrendSense</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="auth-subtitle">Sign in or create your account to access market intelligence and AI planning.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="auth-shell">', unsafe_allow_html=True)
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
    users = _load_users()

    with center_col:
        with login_tab:
            st.caption("Use your existing credentials to continue.")
            with st.form("login_form", clear_on_submit=False):
                login_username = st.text_input("Username")
                login_password = st.text_input("Password", type="password")
                login_clicked = st.form_submit_button("Login", type="primary", use_container_width=True)

            if login_clicked:
                user_key = login_username.strip().lower()
                user_record = users.get(user_key)
                if not user_record:
                    st.error("Account not found. Please sign up first.")
                else:
                    is_valid = _verify_password(
                        login_password,
                        user_record.get("salt", ""),
                        user_record.get("password_hash", ""),
                    )
                    if not is_valid:
                        st.error("Invalid username or password.")
                    else:
                        st.session_state.authenticated = True
                        st.session_state.current_user = user_record.get("username", login_username.strip())
                        st.success("Login successful.")
                        st.rerun()

        with signup_tab:
            st.caption("Password rules: minimum 8 characters, 1 uppercase letter, and 1 special character.")
            with st.form("signup_form", clear_on_submit=True):
                signup_username = st.text_input("Create Username")
                signup_email = st.text_input("Email")
                signup_password = st.text_input("Create Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                signup_clicked = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if signup_clicked:
                clean_username = signup_username.strip()
                clean_email = signup_email.strip()
                user_key = clean_username.lower()
                password_error = _validate_signup_password(signup_password)

                if not clean_username:
                    st.error("Username is required.")
                elif user_key in users:
                    st.error("Username already exists. Please choose a different username.")
                elif password_error is not None:
                    st.error(password_error)
                elif signup_password != confirm_password:
                    st.error("Passwords do not match.")
                elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", clean_email):
                    st.error("Please enter a valid email address.")
                else:
                    salt_hex, hash_hex = _hash_password(signup_password)
                    users[user_key] = {
                        "username": clean_username,
                        "email": clean_email,
                        "salt": salt_hex,
                        "password_hash": hash_hex,
                    }
                    _save_users(users)
                    st.success("Account created. You can now log in from the Login tab.")

        st.markdown("</div>", unsafe_allow_html=True)

    return bool(st.session_state.authenticated)


def _safe_report_filename(category: str) -> str:
    safe_category = re.sub(r"[^a-zA-Z0-9_-]+", "_", category.strip()).strip("_")
    return f"{(safe_category.lower() or 'category')}_trend_report.json"


def load_report_data(file_path: Path) -> list[dict[str, Any]]:
    """Load report rows from local JSON; return empty list on failure."""
    if not file_path.exists():
        return []

    try:
        with file_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError:
        st.warning("The report JSON appears invalid. Please regenerate the report file.")
        return []
    except OSError as exc:
        st.warning(f"Could not read report file: {exc}")
        return []

    if not isinstance(payload, list) or not payload:
        st.warning("No product data available in the report.")
        return []

    return payload


def load_report_metadata(file_path: Path) -> dict[str, Any]:
    """Load the sidecar metadata written by the research backend."""
    metadata_path = file_path.with_name(file_path.name.replace("_trend_report.json", "_trend_metadata.json"))
    if not metadata_path.exists():
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def run_research(category: str, provider: str) -> tuple[list[dict[str, Any]], Path | None]:
    """Run research_graph.py and load its JSON output for shared tab usage."""
    if not category.strip():
        st.warning("Please enter an e-commerce category before starting research.")
        return [], None

    if not RESEARCH_SCRIPT_PATH.exists():
        st.warning(f"Research script not found at: {RESEARCH_SCRIPT_PATH}")
        return [], None

    with st.spinner("Running TrendSense research pipeline..."):
        result = subprocess.run(
            [sys.executable, str(RESEARCH_SCRIPT_PATH), category.strip(), "--provider", provider],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            check=False,
        )

    if result.returncode != 0:
        error_message = (result.stderr or result.stdout or "Unknown runtime error").strip()
        st.warning("Research failed. Check API keys, model setup, and connectivity.")
        st.code(error_message[:3000], language="bash")
        return [], None

    output_path = Path("output") / _safe_report_filename(category)
    rows = load_report_data(output_path)
    metadata = load_report_metadata(output_path)
    if metadata:
        st.session_state.category_profile = metadata.get("category_profile", {}) if isinstance(metadata.get("category_profile"), dict) else {}
        st.session_state.research_category = str(metadata.get("category") or category)
    else:
        st.session_state.research_category = category
    return rows, output_path


def to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize report rows into consistent dashboard columns."""
    frame = pd.DataFrame(rows)
    expected_columns = {
        "product_name": "Unknown Product",
        "success_score": 0.0,
        "demand_score": 0.0,
        "sentiment_score": 50.0,
        "amazon_avg_price": 0.0,
        "aliexpress_avg_price": 0.0,
        "shipping_days": 0.0,
        "seo_description": "No SEO description available.",
    }

    for col_name, default_value in expected_columns.items():
        if col_name not in frame.columns:
            frame[col_name] = default_value

    numeric_columns = [
        "success_score",
        "demand_score",
        "sentiment_score",
        "amazon_avg_price",
        "aliexpress_avg_price",
        "shipping_days",
    ]
    for col_name in numeric_columns:
        frame[col_name] = pd.to_numeric(frame[col_name], errors="coerce").fillna(0.0)

    # Profit margin for card-level financial insight.
    frame["profit_margin_pct"] = frame.apply(
        lambda row: ((row["amazon_avg_price"] - row["aliexpress_avg_price"]) / row["amazon_avg_price"] * 100)
        if row["amazon_avg_price"] > 0
        else 0.0,
        axis=1,
    )

    frame = frame.sort_values(by="success_score", ascending=False)
    return frame


def render_header_metrics(frame: pd.DataFrame) -> None:
    """Render top-row KPIs for quick executive summary."""
    total_products = int(frame.shape[0])

    if total_products > 0:
        top_row = frame.loc[frame["success_score"].idxmax()]
        top_product_name = str(top_row["product_name"])
        average_score = float(frame["success_score"].mean())
    else:
        top_product_name = "N/A"
        average_score = 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Top Rated Product", top_product_name)
    col2.metric("Average Success Score", f"{average_score:.2f}")
    col3.metric("Total Products Analyzed", f"{total_products}")


def render_score_chart(frame: pd.DataFrame) -> None:
    """Horizontal Plotly chart of product vs success score with score-based color."""
    if frame.empty:
        st.info("No products available to visualize yet.")
        return

    chart = px.bar(
        frame,
        x="success_score",
        y="product_name",
        orientation="h",
        color="success_score",
        color_continuous_scale="Viridis",
        labels={"product_name": "Product Name", "success_score": "Success Score"},
        title="Product Opportunity Scores",
    )
    chart.update_layout(height=500, coloraxis_colorbar_title="Score")
    st.plotly_chart(chart, use_container_width=True, key="market_success_score_chart")


def render_product_cards(frame: pd.DataFrame) -> None:
    """Expandable cards with financial, logistics, and SEO intelligence."""
    if frame.empty:
        return

    st.subheader("Detailed Intelligence Cards")

    # Show highest-scoring products first in cards.
    display_frame = frame.sort_values(by="success_score", ascending=False)

    for card_idx, (_, row) in enumerate(display_frame.iterrows()):
        product_name = str(row["product_name"])
        with st.expander(f"{product_name}  |  Success Score: {row['success_score']:.2f}"):
            financial_col, logistics_col, opinion_col = st.columns(3)

            with financial_col:
                st.markdown("**Financials**")
                st.write(f"Amazon Price: ${row['amazon_avg_price']:.2f}")
                st.write(f"AliExpress Price: ${row['aliexpress_avg_price']:.2f}")
                st.write(f"Profit Margin: {row['profit_margin_pct']:.2f}%")

            with logistics_col:
                st.markdown("**Logistics**")
                st.write(f"Demand Score: {row['demand_score']:.2f}")
                st.write(f"Shipping Days: {row['shipping_days']:.2f}")

            with opinion_col:
                st.markdown("**Public Opinion**")
                sentiment_score = float(row.get("sentiment_score", 50.0))
                st.write(f"Sentiment Score: {sentiment_score:.2f}/100")

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=sentiment_score,
                        number={"suffix": "/100"},
                        title={"text": "Public Opinion"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#2E86DE"},
                            "steps": [
                                {"range": [0, 35], "color": "#F5B7B1"},
                                {"range": [35, 70], "color": "#F9E79F"},
                                {"range": [70, 100], "color": "#ABEBC6"},
                            ],
                        },
                    )
                )
                gauge.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
                safe_product = re.sub(r"[^a-zA-Z0-9_-]+", "_", product_name).strip("_") or "product"
                st.plotly_chart(
                    gauge,
                    use_container_width=True,
                    key=f"public_opinion_gauge_{safe_product}_{card_idx}",
                )

            st.markdown("**AI-Generated SEO Description**")
            st.markdown('<div class="seo-box">', unsafe_allow_html=True)
            st.code(str(row["seo_description"]), language="markdown")
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="small-note">Copy feel: use the code-block copy button in the top-right corner.</div>',
                unsafe_allow_html=True,
            )


def _escape_md(text: Any) -> str:
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def _get_ai_llm(provider: str):
    """Return the best available LLM for the ad planner."""
    if provider == "Cloud (Groq)":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing.")

        model_candidates = [
            os.getenv("GROQ_MODEL", "").strip(),
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
        ]

        last_error: Exception | None = None
        for model_name in [name for name in model_candidates if name]:
            try:
                llm = ChatGroq(model=model_name, api_key=api_key, temperature=0.4)
                llm.invoke("Respond with exactly: ok")
                return llm
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(f"No supported Groq model is available. Last error: {last_error}")

    if ChatOllama is None:
        raise RuntimeError("langchain-ollama is not installed.")

    for model_name in ("qwen2.5:7b", "qwen2.5:7b-instruct"):
        try:
            llm = ChatOllama(model=model_name, temperature=0.4)
            llm.invoke("Respond with exactly: ok")
            return llm
        except Exception:
            continue

    raise RuntimeError("Local Ollama model not found. Pull qwen2.5:7b or qwen2.5:7b-instruct.")


def build_export_markdown(frame: pd.DataFrame) -> str:
    """Build a clean Markdown project report from the current session state."""
    category = st.session_state.get("research_category") or str(st.session_state.get("category_profile", {}).get("category", "Project"))
    category_profile = st.session_state.get("category_profile", {}) or {}
    top_frame = frame.head(5).copy() if not frame.empty else frame

    lines: list[str] = []
    lines.append(f"# TrendSense Intelligence Report: {category}")
    lines.append("")
    lines.append("You can paste this Markdown into any MD-to-PDF converter for your final project file.")
    lines.append("")
    lines.append("## Executive Summary")
    if frame.empty:
        lines.append("No research data is currently loaded in this session.")
    else:
        avg_success = float(frame["success_score"].mean())
        top_product = str(frame.iloc[0]["product_name"])
        lines.append(
            f"The current market scan identified {len(frame)} product opportunities. "
            f"The leading product is **{top_product}** with an average success score baseline of **{avg_success:.2f}** across the session. "
            "Use the detailed sections below to review demand, price fit, sentiment, and marketing direction."
        )
        if category_profile:
            lines.append("")
            lines.append("### Category Profile")
            lines.append(f"- Minimum Viable Price: ${float(category_profile.get('minimum_viable_price', 0.0)):.2f}")
            lines.append(f"- Maximum Viable Price: ${float(category_profile.get('maximum_viable_price', 0.0)):.2f}")
            forbidden = category_profile.get("forbidden_keywords", [])
            forbidden_text = ", ".join(str(item) for item in forbidden) if forbidden else "None"
            lines.append(f"- Forbidden Keywords: {forbidden_text}")

    lines.append("")
    lines.append("## Deep Dive")
    if top_frame.empty:
        lines.append("No product details available.")
    else:
        for _, row in top_frame.iterrows():
            product_name = str(row.get("product_name", "Unknown Product"))
            success_score = float(row.get("success_score", 0.0))
            demand_score = float(row.get("demand_score", 0.0))
            margin_score = float(row.get("margin_score", 0.0))
            reliability_score = float(row.get("supply_reliability_score", 0.0))
            sentiment_score = float(row.get("sentiment_score", 50.0))
            amazon_price = float(row.get("amazon_avg_price", 0.0))
            aliexpress_price = float(row.get("aliexpress_avg_price", 0.0))
            shipping_days = float(row.get("shipping_days", 0.0))
            seo_description = _escape_md(row.get("seo_description", ""))

            lines.append(f"### {product_name}")
            lines.append(f"- Success Score: {success_score:.2f}")
            lines.append(f"- Demand Score: {demand_score:.2f}")
            lines.append(f"- Margin Score: {margin_score:.2f}")
            lines.append(f"- Reliability Score: {reliability_score:.2f}")
            lines.append(f"- Sentiment Score: {sentiment_score:.2f}")
            lines.append(f"- Amazon Avg Price: ${amazon_price:.2f}")
            lines.append(f"- AliExpress Avg Price: ${aliexpress_price:.2f}")
            lines.append(f"- Shipping Days: {shipping_days:.2f}")
            lines.append(f"- SEO Description: {seo_description}")
            lines.append("")
            lines.append("#### Success Score Breakdown")
            lines.append(
                f"Success Score = (0.4 × {demand_score:.2f}) + (0.3 × {margin_score:.2f}) + (0.2 × {reliability_score:.2f}) + (0.1 × {sentiment_score:.2f})"
            )
            lines.append("")

    lines.append("## AI Strategy")
    if not st.session_state.ai_outputs:
        lines.append("No Ollama ad plans have been generated in this session yet.")
    else:
        for product_name, payload in st.session_state.ai_outputs.items():
            lines.append(f"### {product_name}")
            lines.append("#### 150-Word Product Description")
            lines.append(_escape_md(payload.get("description", "")))
            lines.append("")
            lines.append("#### 3-Phase Plan")
            lines.append(_escape_md(payload.get("ad_plan", "")))
            lines.append("")
            lines.append("#### Viral Hooks")
            lines.append(_escape_md(payload.get("hooks", "")))
            lines.append("")

    lines.append("## Appendix: Technical Reasoning")
    lines.append("### Category Profile Logic")
    if category_profile:
        lines.append("Raw JSON used by the Discovery and Deep Research nodes:")
        lines.append("```json")
        lines.append(json.dumps(category_profile, indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append("No category profile metadata was available in the current session.")

    lines.append("")
    lines.append("### Sentiment Evidence")
    if frame.empty:
        lines.append("No sentiment evidence is available because no research data is loaded.")
    else:
        for _, row in frame.head(5).iterrows():
            product_name = str(row.get("product_name", "Unknown Product"))
            sentiment_score = float(row.get("sentiment_score", 50.0))
            sentiment_notes = _escape_md(row.get("sentiment_notes", ""))
            lines.append(f"#### {product_name}")
            lines.append(f"- Sentiment Score: {sentiment_score:.2f}")
            lines.append("- Sentiment Notes:")
            lines.append(sentiment_notes if sentiment_notes.strip() else "No explicit sentiment evidence was stored for this product.")
            lines.append("")

    lines.append("### Developer Note")
    lines.append(
        "Intelligence was generated via **On-Device Quantized Inference** using **Qwen 2.5 7B** for local analysis, sanitization, and sentiment reasoning where applicable."
    )

    return "\n".join(lines).strip() + "\n"


@st.cache_data(ttl=60, show_spinner=False)
def check_tavily_api_status() -> tuple[bool, str]:
    """Quick health check for Tavily connectivity shown in sidebar."""
    if not os.getenv("TAVILY_API_KEY"):
        return False, "TAVILY_API_KEY missing"

    try:
        tool = TavilySearchResults(max_results=1)
        tool.invoke({"query": "wireless earbuds buy now"})
        return True, "Connected"
    except Exception as exc:
        return False, str(exc)[:140]


def render_sidebar(default_path: Path) -> tuple[Path, str]:
    """Sidebar controls for report path, methodology context, and refresh trigger."""
    st.sidebar.header("TrendSense Controls")
    st.sidebar.caption(f"Signed in as: {st.session_state.current_user or 'User'}")

    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user = ""
        st.toast("Logged out successfully.", icon="👋")
        st.rerun()

    st.sidebar.markdown("---")

    selected_provider = st.sidebar.selectbox(
        "LLM Intelligence Layer",
        options=PROVIDER_OPTIONS,
        index=PROVIDER_OPTIONS.index(st.session_state.llm_provider),
    )
    st.session_state.llm_provider = selected_provider

    tavily_ok, tavily_message = check_tavily_api_status()
    if tavily_ok:
        st.sidebar.success("Tavily API Status: Connected")
    else:
        st.sidebar.error(f"Tavily API Status: Not Connected ({tavily_message})")

    report_path_text = st.sidebar.text_input("JSON Report Path", value=st.session_state.research_file_path or str(default_path))

    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.write(
        "TrendSense uses a Weighted Decision Matrix to prioritize products based on "
        "multi-factor intelligence."
    )
    st.sidebar.markdown(
        """
        - Demand: **50%**
        - Margin: **35%**
        - Reliability: **15%**
        """
    )

    if st.sidebar.button("Refresh Data"):
        # UI-only interaction for now; no backend state mutation is required.
        st.toast("Data refresh requested. Reloading dashboard...", icon="🔄")
        st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
        st.session_state.research_data = []
        st.session_state.research_results = []
        st.session_state.category_profile = {}
        st.session_state.research_category = ""
        st.session_state.research_file_path = str(default_path)
        st.session_state.ai_outputs = {}
        st.session_state.llm_provider = PROVIDER_OPTIONS[0]
        st.toast("All data cleared. Starting fresh...", icon="✨")
        st.rerun()

    return Path(report_path_text), selected_provider


def render_market_research_tab(selected_provider: str) -> None:
    """Tab 1: Market Intelligence - trigger research, persist JSON results, and visualize KPIs."""
    st.subheader("Market Intelligence")

    category = st.text_input(
        "Enter E-Commerce Category",
        value="Car Accessories",
        help="Example: beauty products, smart home productivity, pet accessories, iphone 17 pro",
    )

    start_research = st.button("Start Research", type="primary", use_container_width=True)
    if start_research:
        rows, output_path = run_research(category, selected_provider)
        if rows:
            st.session_state.research_data = rows
            st.session_state.research_results = rows
            if output_path is not None:
                st.session_state.research_file_path = str(output_path)
            st.success(f"✅ Research completed. Loaded {len(rows)} products.")

    st.caption(
        "💡 Tip: Use Start Research for fresh analysis, or load an existing local JSON report path from the sidebar."
    )

    if not st.session_state.research_data:
        st.info("No research data in session yet. Run research or load a local JSON report.")
        return

    frame = to_dataframe(st.session_state.research_data)
    render_header_metrics(frame)
    st.markdown("---")
    render_score_chart(frame)
    render_product_cards(frame)


def generate_local_ai_content(product_row: pd.Series) -> dict[str, str]:
    """Generate product description, 3-phase ad plan, and viral hooks using the selected provider."""
    provider = st.session_state.get("llm_provider", "Cloud (Groq)")
    try:
        llm = _get_ai_llm(provider)
    except Exception:
        if provider != "Cloud (Groq)":
            llm = _get_ai_llm("Cloud (Groq)")
        else:
            raise

    product_name = str(product_row["product_name"])
    source_description = str(product_row.get("seo_description", "")).strip()
    amazon_price = float(product_row['amazon_avg_price'])
    aliexpress_price = float(product_row['aliexpress_avg_price'])
    margin = ((amazon_price - aliexpress_price) / amazon_price * 100) if amazon_price > 0 else 0

    consistency_prompt = f"""You are a strict data consistency validator for e-commerce.

Product Name: {product_name}
Provided Description: {source_description}

Task:
- Determine whether the provided description matches the product's actual purpose.
- Example mismatch: a phone case described as a skincare kit.

If the description does not match, return exactly:
Data Inconsistency Detected

If it matches, return exactly:
OK
""".strip()

    consistency_result = str(llm.invoke(consistency_prompt).content).strip()
    consistency_result = re.sub(r"^```(?:text)?\s*|\s*```$", "", consistency_result, flags=re.IGNORECASE | re.DOTALL).strip()
    if "Data Inconsistency Detected" in consistency_result:
        raise ValueError("Data Inconsistency Detected")

    # 150-word product description
    description_prompt = f"""You are a high-converting e-commerce copywriter. Create exactly a 150-word product description for {product_name}.

Key metrics:
- Profit margin: {margin:.1f}%
- Amazon price: ${amazon_price:.2f}
- Demand score: {float(product_row['demand_score']):.2f}

Focus on benefits, urgency, and trust signals. Be persuasive but honest. Target: {product_name} buyers.
Word count: exactly 150 words. Return ONLY the description, no metadata.""".strip()

    # 3-Phase advertising plan with phases
    ad_plan_prompt = f"""Create a 3-phase advertising strategy for {product_name}:

Phase 1: AWARENESS - How to introduce the product to new audiences
Phase 2: CONSIDERATION - How to build desire and justify the purchase
Phase 3: CONVERSION - How to drive sales and urgency

Profit margin: {margin:.1f}%
Target audience: {product_name} enthusiasts and buyers.

Format each phase as:
**Phase X: [PHASE_NAME]**
- Strategy point 1
- Strategy point 2

Keep each phase 2-3 bullet points. Be specific and tactical.""".strip()

    # 5 viral social media hooks
    hooks_prompt = f"""Generate exactly 5 short, viral-worthy social media hooks (for TikTok/Instagram) for {product_name}.

Each hook should be 1 sentence, catchy, and drive engagement. Include:
- 1 "Problem/Solution" hook
- 1 "Emotion/Curiosity" hook
- 1 "FOMO/Scarcity" hook
- 1 "Value/ROI" hook
- 1 "Trend/Viral" hook

Format as numbered list. Return ONLY the hooks.""".strip()

    description = str(llm.invoke(description_prompt).content).strip()
    ad_plan = str(llm.invoke(ad_plan_prompt).content).strip()
    hooks = str(llm.invoke(hooks_prompt).content).strip()
    
    return {
        "description": description,
        "ad_plan": ad_plan,
        "hooks": hooks,
    }


def render_ai_content_tab() -> None:
    """Tab 2: AI Ad Planner - description, 3-phase plan, and viral hooks using the selected provider."""
    st.subheader("AI Ad Planner")

    if not st.session_state.research_data:
        st.info("Run Market Intelligence first to unlock AI content generation.")
        return

    frame = to_dataframe(st.session_state.research_data)
    product_options = frame["product_name"].tolist()
    selected_product_name = st.selectbox("Select Product for Strategy", options=product_options, key="ai_product_select")

    col1, col2 = st.columns([1, 1])
    with col1:
        generate_clicked = st.button("🚀 Generate Strategy", type="primary", use_container_width=True)
    
    if generate_clicked:
        selected_row = frame.loc[frame["product_name"] == selected_product_name].iloc[0]
        try:
            provider_label = st.session_state.get("llm_provider", "Cloud (Groq)")
            with st.spinner(f"Generating AI strategy via {provider_label}..."):
                content = generate_local_ai_content(selected_row)
            st.session_state.ai_outputs[selected_product_name] = content
            st.success(f"✅ Strategy generated for {selected_product_name}")
        except Exception as exc:
            if "Data Inconsistency Detected" in str(exc):
                st.error("Data Inconsistency Detected")
            else:
                st.error("Could not generate content with the selected AI provider.")
                st.code(str(exc), language="bash")

    if selected_product_name in st.session_state.ai_outputs:
        generated = st.session_state.ai_outputs[selected_product_name]
        
        # 150-word description
        st.markdown("### 📝 High-Conversion Product Description (150 words)")
        st.info(generated["description"])
        
        # 3-Phase advertising plan
        st.markdown("### 📢 3-Phase Advertising Plan")
        st.markdown(generated["ad_plan"])
        
        # 5 viral hooks
        st.markdown("### 🎯 5 Viral TikTok/Instagram Hooks")
        st.markdown(generated["hooks"])
        
        # Copy-to-clipboard helper
        st.markdown("---")
        st.caption("💡 Tip: Select and copy any text above to use in your marketing tools.")
    else:
        st.caption("👈 Select a product and click 'Generate Strategy' to create AI-powered marketing content.")


def render_comparison_tab() -> None:
    """Tab 3: Radar chart comparing top 3 products, plus full comparison table."""
    st.subheader("Compare SKU")

    if not st.session_state.research_data:
        st.info("Run Market Intelligence first to compare products.")
        return

    frame = to_dataframe(st.session_state.research_data)
    
    # Comparison table
    st.markdown("### 📊 Product Comparison Table")
    comparison = frame[
        [
            "product_name",
            "success_score",
            "demand_score",
            "amazon_avg_price",
            "aliexpress_avg_price",
            "profit_margin_pct",
        ]
    ].rename(
        columns={
            "product_name": "Product Name",
            "success_score": "Success Score",
            "demand_score": "Demand",
            "amazon_avg_price": "Amazon Price",
            "aliexpress_avg_price": "AliExpress Price",
            "profit_margin_pct": "Margin %",
        }
    )

    st.dataframe(comparison, use_container_width=True, hide_index=True)

    # Radar chart for top 3 products
    st.markdown("### 🎯 Radar Chart: Top 3 Products (Demand vs. Margin vs. Reliability)")
    
    if len(frame) >= 3:
        top_3 = frame.head(3).copy()
        
        # Extract metrics for radar (normalize to 0-100 scale for visibility)
        radar_data = []
        for _, row in top_3.iterrows():
            radar_data.append({
                "Product": str(row["product_name"]),
                "Demand": float(row["demand_score"]) * 10,  # Scale up for visibility
                "Margin": float(row["profit_margin_pct"]),
                "Reliability": float(row["success_score"]) * 10,  # Scale up
            })
        
        radar_df = pd.DataFrame(radar_data)
        
        fig = px.line_polar(
            radar_df,
            r="Demand",
            theta=["Demand", "Margin", "Reliability"],
            color="Product",
            title="Top 3 Products: Performance Radar",
            markers=True,
            line_close=True,
        )
        
        # Add additional dimensions by creating custom radar
        figures = []
        for product in radar_df["Product"].unique():
            product_row = radar_df[radar_df["Product"] == product].iloc[0]
            fig_product = px.line_polar(
                r=[product_row["Demand"], product_row["Margin"], product_row["Reliability"]],
                theta=["Demand Score", "Margin %", "Reliability Score"],
                title=f"{product}",
                line_close=True,
                markers=True,
            )
            figures.append(fig_product)
        
        # Display radar charts in columns
        if len(top_3) >= 3:
            col1, col2, col3 = st.columns(3)
            
            for idx, (_, row) in enumerate(top_3.iterrows()):
                if idx == 0:
                    col = col1
                elif idx == 1:
                    col = col2
                else:
                    col = col3
                
                with col:
                    product_name = str(row["product_name"])
                    demand = float(row["demand_score"])
                    margin = float(row["profit_margin_pct"])
                    reliability = float(row["success_score"])
                    
                    fig_single = px.line_polar(
                        r=[demand, margin, reliability],
                        theta=["Demand", "Margin %", "Reliability"],
                        title=f"📈 {product_name}",
                        line_close=True,
                        markers=True,
                    )
                    fig_single.update_layout(height=400)
                    safe_product = re.sub(r"[^a-zA-Z0-9_-]+", "_", product_name).strip("_") or "product"
                    st.plotly_chart(
                        fig_single,
                        use_container_width=True,
                        key=f"compare_radar_{safe_product}_{idx}",
                    )
    else:
        st.info(f"Need at least 3 products to display radar chart. Current: {len(frame)} product(s).")

    # Bar chart selector
    st.markdown("### 📉 Detailed Metric Comparison")
    metric_choice = st.selectbox(
        "Select Metric to Visualize",
        options=["Success Score", "Demand", "Amazon Price", "AliExpress Price", "Margin %"],
    )
    
    chart = px.bar(
        comparison.sort_values(metric_choice, ascending=False),
        x="Product Name",
        y=metric_choice,
        color=metric_choice,
        color_continuous_scale="RdYlGn",
        title=f"Product Comparison by {metric_choice}",
    )
    chart.update_layout(xaxis_tickangle=-25, height=500)
    safe_metric = re.sub(r"[^a-zA-Z0-9_-]+", "_", metric_choice).strip("_") or "metric"
    st.plotly_chart(chart, use_container_width=True, key=f"compare_metric_bar_{safe_metric}")


def main() -> None:
    initialize_session_state()

    if not st.session_state.authenticated:
        render_auth_pages()
        return

    st.title("TrendSense: AI-Driven E-Commerce Research Dashboard")
    st.caption("MCA Major Project Frontend")

    selected_report_path, selected_provider = render_sidebar(DEFAULT_REPORT_PATH)

    # Load user-selected local report into session only if no in-memory results exist.
    if not st.session_state.research_data:
        rows = load_report_data(selected_report_path)
        if rows:
            st.session_state.research_data = rows
            st.session_state.research_results = rows
            st.session_state.research_file_path = str(selected_report_path)
            metadata = load_report_metadata(selected_report_path)
            if metadata:
                st.session_state.category_profile = metadata.get("category_profile", {}) if isinstance(metadata.get("category_profile"), dict) else {}
                st.session_state.research_category = str(metadata.get("category") or "")

    tab1, tab2, tab3 = st.tabs([
        "🔍 Market Intelligence",
        "📢 AI Ad Planner",
        "⚖️ Compare SKU",
    ])

    with tab1:
        render_market_research_tab(selected_provider)
    with tab2:
        render_ai_content_tab()
    with tab3:
        render_comparison_tab()

    export_frame = to_dataframe(st.session_state.research_data) if st.session_state.research_data else pd.DataFrame()
    markdown_report = build_export_markdown(export_frame)
    export_filename = f"trendSense_major_project_report_{re.sub(r'[^a-zA-Z0-9_-]+', '_', (st.session_state.get('research_category') or 'project')).strip('_').lower() or 'project'}.md"
    st.sidebar.download_button(
        label="📥 Export Major Project Report",
        data=markdown_report,
        file_name=export_filename,
        mime="text/markdown",
        use_container_width=True,
    )
    st.sidebar.caption("You can paste this Markdown into any MD-to-PDF converter for your final project file.")


if __name__ == "__main__":
    main()
