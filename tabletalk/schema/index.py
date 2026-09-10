"""Schema cards and table retrieval.

A card is the compact description of one table the model sees: columns with types, foreign
keys, and three sample rows. Small schemas are passed whole. Only when the full schema would
not fit the token budget are cards embedded and the closest ones to the question selected,
plus their one-hop foreign-key neighbours so joins stay possible.
"""

import sqlite3
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

EmbedFn = Callable[[list[str]], np.ndarray]


@dataclass
class TableCard:
    name: str
    columns: list[tuple[str, str, bool]]  # (name, type, is_pk)
    foreign_keys: list[tuple[str, str, str]]  # (column, ref_table, ref_column)
    sample_rows: list[list]
    row_count: int

    def render(self) -> str:
        cols = ", ".join(
            f"{c} {t}{' PK' if pk else ''}".strip() for c, t, pk in self.columns
        )
        lines = [f"TABLE {self.name} ({cols})  -- {self.row_count} rows"]
        for col, ref_t, ref_c in self.foreign_keys:
            lines.append(f"  FK {self.name}.{col} -> {ref_t}.{ref_c}")
        if self.sample_rows:
            names = [c for c, _, _ in self.columns]
            lines.append("  sample: " + " | ".join(names))
            for row in self.sample_rows:
                lines.append("          " + " | ".join(_short(v) for v in row))
        return "\n".join(lines)

    def summary(self) -> str:
        """Short text used for embedding: table name plus column names."""
        return f"{self.name}: " + ", ".join(c for c, _, _ in self.columns)


def _short(value, width: int = 40) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_schema(db_path: str, sample_rows: int = 3) -> list[TableCard]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        cards = []
        for name in names:
            cols = [
                (r[1], r[2] or "", bool(r[5]))
                for r in conn.execute(f"PRAGMA table_info({_quote(name)})")
            ]
            fks = [
                (r[3], r[2], r[4] or "")
                for r in conn.execute(f"PRAGMA foreign_key_list({_quote(name)})")
            ]
            try:
                count = conn.execute(f"SELECT count(*) FROM {_quote(name)}").fetchone()[0]
                rows = [
                    list(r)
                    for r in conn.execute(f"SELECT * FROM {_quote(name)} LIMIT {sample_rows}")
                ]
            except sqlite3.DatabaseError:
                count, rows = 0, []
            cards.append(TableCard(name, cols, fks, rows, count))
        return cards
    finally:
        conn.close()


def render_schema(cards: list[TableCard]) -> str:
    return "\n\n".join(c.render() for c in cards)


def estimate_tokens(text: str) -> int:
    # good enough for a budget check; real tokenizers land within ~20% on schema text
    return len(text) // 4 + 1


@dataclass
class SchemaSelection:
    cards: list[TableCard]
    retrieved: bool
    total_tables: int
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return render_schema(self.cards)


_embedder: EmbedFn | None = None


def default_embedder() -> EmbedFn:
    """multilingual-e5-small, loaded on first use. Query/passage prefixes are what the model
    was trained with; without them scores are noticeably worse."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")

        def embed(texts: list[str]) -> np.ndarray:
            return model.encode(texts, normalize_embeddings=True)

        _embedder = embed
    return _embedder


def select_tables(
    cards: list[TableCard],
    question: str,
    token_budget: int,
    embed: EmbedFn | None = None,
) -> SchemaSelection:
    full = render_schema(cards)
    if estimate_tokens(full) <= token_budget or len(cards) <= 2:
        return SchemaSelection(cards=cards, retrieved=False, total_tables=len(cards))

    embed = embed or default_embedder()
    vectors = embed(["passage: " + c.summary() for c in cards])
    q = embed(["query: " + question])[0]
    scores = vectors @ q
    order = np.argsort(-scores)

    chosen: list[TableCard] = []
    chosen_names: set[str] = set()
    by_name = {c.name: c for c in cards}
    used = 0
    for idx in order:
        card = cards[int(idx)]
        if card.name in chosen_names:
            continue
        cost = estimate_tokens(card.render())
        if used + cost > token_budget and chosen:
            break
        chosen.append(card)
        chosen_names.add(card.name)
        used += cost
        # one-hop FK neighbours make joins possible even if their own score is low
        for _, ref_table, _ in card.foreign_keys:
            neighbour = by_name.get(ref_table)
            if neighbour and neighbour.name not in chosen_names:
                n_cost = estimate_tokens(neighbour.render())
                if used + n_cost <= token_budget:
                    chosen.append(neighbour)
                    chosen_names.add(neighbour.name)
                    used += n_cost
    chosen.sort(key=lambda c: c.name)
    return SchemaSelection(
        cards=chosen,
        retrieved=True,
        total_tables=len(cards),
        scores={c.name: float(s) for c, s in zip(cards, scores)},
    )
