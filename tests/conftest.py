"""Shared pytest fixtures for the chunking engine test suite."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from chunking_engine.chunkers import ChunkingConfig


# ── Common text fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def general_text() -> str:
    return (
        "Introduction to Machine Learning\n\n"
        "Machine learning is a branch of artificial intelligence that enables "
        "systems to learn from data and improve their performance over time "
        "without being explicitly programmed.\n\n"
        "There are three main types of machine learning:\n"
        "supervised learning, unsupervised learning, and reinforcement learning.\n\n"
        "Supervised learning uses labelled training data to learn a mapping "
        "from inputs to outputs. Common algorithms include linear regression, "
        "decision trees, and neural networks.\n\n"
        "Unsupervised learning discovers hidden patterns in data without labels. "
        "Clustering and dimensionality reduction are typical use cases."
    )


@pytest.fixture
def structured_document() -> str:
    return (
        "Incident Report — INC-2024-0042\n"
        "Priority: High\n\n"
        "Overview:\n"
        "A critical service outage affected the payment processing pipeline "
        "for approximately 45 minutes on 2024-03-15.\n\n"
        "Description:\n"
        "The root cause was a deadlock in the database connection pool "
        "triggered by a surge in concurrent transactions after a marketing campaign.\n\n"
        "Findings:\n"
        "- Connection pool size was set to 10, insufficient for peak load\n"
        "- No circuit breaker was in place for downstream payment service\n"
        "- Alerting threshold was set too conservatively\n\n"
        "Recommendations:\n"
        "1. Increase connection pool to 50 and add auto-scaling\n"
        "2. Implement circuit breaker pattern\n"
        "3. Lower alerting thresholds and add predictive alerts\n\n"
        "References:\n"
        "See post-mortem document PM-2024-0042 for full timeline."
    )


@pytest.fixture
def transcript_text() -> str:
    return (
        "00:00:05 Alice: Good morning everyone, let's get started with the sprint review.\n"
        "00:00:12 Bob: Sure. We completed the authentication module this sprint.\n"
        "00:00:45 Alice: Great. What was the main challenge?\n"
        "00:01:02 Bob: Token refresh logic was tricky, but it's done now.\n\n"
        "00:05:30 Carol: Moving on to the deployment pipeline — we've reduced build time by 40%.\n"
        "00:05:55 Alice: That's impressive. What did you change?\n"
        "00:06:10 Carol: We parallelised the test stages and added caching for dependencies.\n\n"
        "00:12:00 Dave: For next sprint, I propose we tackle the search feature.\n"
        "00:12:20 Alice: Agreed. Let's add that to the backlog."
    )


@pytest.fixture
def python_code() -> str:
    return (
        "import math\n\n"
        "def calculate_area(radius: float) -> float:\n"
        "    \"\"\"Return the area of a circle.\"\"\"\n"
        "    return math.pi * radius ** 2\n\n"
        "def calculate_circumference(radius: float) -> float:\n"
        "    \"\"\"Return the circumference of a circle.\"\"\"\n"
        "    return 2 * math.pi * radius\n\n"
        "class Circle:\n"
        "    def __init__(self, radius: float) -> None:\n"
        "        self.radius = radius\n\n"
        "    def area(self) -> float:\n"
        "        return calculate_area(self.radius)\n\n"
        "    def circumference(self) -> float:\n"
        "        return calculate_circumference(self.radius)\n"
    )


@pytest.fixture
def csv_data() -> str:
    return (
        "ID,Name,Category,Priority,Status,Description\n"
        "ITEM-001,Database Migration,Infrastructure,High,Mandatory,Migrate from v5 to v8 before EOY\n"
        "ITEM-002,API Gateway Update,Security,Critical,Mandatory,Update TLS certificates expiring Q1\n"
        "ITEM-003,Legacy Service Removal,Maintenance,Medium,Optional,Decommission deprecated billing service\n"
        "ITEM-004,Performance Audit,Optimisation,Low,Recommended,Profile and optimise top 10 slow queries\n"
    )


@pytest.fixture
def default_config() -> ChunkingConfig:
    return ChunkingConfig(chunk_size=500, chunk_overlap=50)
