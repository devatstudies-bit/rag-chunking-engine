"""
Sample Python module — demonstrates code-aware chunking.

This module provides a simple inventory management system used as a
demo document for the code-aware chunking strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Product:
    """Represents a single product in the inventory."""

    sku: str
    name: str
    unit_price: float
    quantity: int = 0
    category: str = "general"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def is_in_stock(self) -> bool:
        return self.quantity > 0

    def total_value(self) -> float:
        return self.unit_price * self.quantity


class InventoryError(Exception):
    """Raised for invalid inventory operations."""


class Inventory:
    """Thread-unsafe in-memory inventory store for demonstration purposes."""

    def __init__(self) -> None:
        self._products: dict[str, Product] = {}

    def add_product(self, product: Product) -> None:
        """Register a new product; raise if SKU already exists."""
        if product.sku in self._products:
            raise InventoryError(f"SKU '{product.sku}' already registered")
        self._products[product.sku] = product
        logger.info("product_added", sku=product.sku, name=product.name)

    def remove_product(self, sku: str) -> Product:
        """Remove and return a product by SKU."""
        if sku not in self._products:
            raise InventoryError(f"SKU '{sku}' not found")
        product = self._products.pop(sku)
        logger.info("product_removed", sku=sku)
        return product

    def restock(self, sku: str, quantity: int) -> int:
        """Add *quantity* units to a product; return new quantity."""
        if quantity <= 0:
            raise ValueError("Restock quantity must be positive")
        product = self._get_or_raise(sku)
        product.quantity += quantity
        logger.info("product_restocked", sku=sku, added=quantity, total=product.quantity)
        return product.quantity

    def sell(self, sku: str, quantity: int) -> float:
        """Sell *quantity* units; return total revenue; raise if insufficient stock."""
        if quantity <= 0:
            raise ValueError("Sell quantity must be positive")
        product = self._get_or_raise(sku)
        if product.quantity < quantity:
            raise InventoryError(
                f"Insufficient stock for '{sku}': requested {quantity}, available {product.quantity}"
            )
        product.quantity -= quantity
        revenue = product.unit_price * quantity
        logger.info("product_sold", sku=sku, quantity=quantity, revenue=revenue)
        return revenue

    def total_inventory_value(self) -> float:
        """Return the sum of (unit_price * quantity) for all products."""
        return sum(p.total_value() for p in self._products.values())

    def low_stock_report(self, threshold: int = 5) -> list[Product]:
        """Return products with quantity at or below *threshold*."""
        return [p for p in self._products.values() if p.quantity <= threshold]

    def _get_or_raise(self, sku: str) -> Product:
        if sku not in self._products:
            raise InventoryError(f"SKU '{sku}' not found")
        return self._products[sku]


def build_sample_inventory() -> Inventory:
    """Populate and return a sample inventory for testing."""
    inv = Inventory()
    products = [
        Product(sku="WIDGET-001", name="Blue Widget", unit_price=9.99, quantity=100, category="widgets"),
        Product(sku="WIDGET-002", name="Red Widget", unit_price=12.50, quantity=3, category="widgets"),
        Product(sku="GADGET-001", name="Smart Gadget", unit_price=49.99, quantity=25, category="gadgets"),
        Product(sku="GADGET-002", name="Mini Gadget", unit_price=19.99, quantity=0, category="gadgets"),
    ]
    for p in products:
        inv.add_product(p)
    return inv


def format_inventory_report(inventory: Inventory) -> str:
    """Return a human-readable inventory summary."""
    lines = ["=== Inventory Report ==="]
    for sku, product in inventory._products.items():
        status = "IN STOCK" if product.is_in_stock() else "OUT OF STOCK"
        lines.append(
            f"  [{sku}] {product.name} | Qty: {product.quantity} | "
            f"Price: ${product.unit_price:.2f} | {status}"
        )
    lines.append(f"\nTotal inventory value: ${inventory.total_inventory_value():.2f}")
    return "\n".join(lines)
