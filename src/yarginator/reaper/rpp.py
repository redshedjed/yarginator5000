"""Minimal reader/writer for REAPER's .RPP text format.

An RPP file is a tree of ``<NAME args ...`` blocks closed by ``>``, with one ``KEY values...`` line
per setting. Lines we don't touch keep their original text, so a template round-trips unchanged
(plugin state, base64 blobs and all).
"""
from __future__ import annotations

from typing import Iterator


def tokenize(line: str) -> list[str]:
    """Split on whitespace; a token may be quoted with ", ' or ` (REAPER picks whichever the
    content doesn't contain)."""
    out, i, n = [], 0, len(line)
    while i < n:
        if line[i].isspace():
            i += 1
            continue
        if line[i] in "\"'`":
            q, j = line[i], line.find(line[i], i + 1)
            j = n if j < 0 else j
            out.append(line[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            out.append(line[i:j])
            i = j
    return out


def quote(tok) -> str:
    s = str(tok)
    if s and not any(c.isspace() or c in "\"'`" for c in s):
        return s
    for q in "\"'`":
        if q not in s:
            return f"{q}{s}{q}"
    return '"' + s.replace('"', "'") + '"'


def format_line(*tokens) -> str:
    return " ".join(quote(t) for t in tokens)


class Line:
    __slots__ = ("raw",)

    def __init__(self, raw: str):
        self.raw = raw

    @classmethod
    def of(cls, *tokens) -> "Line":
        return cls(format_line(*tokens))

    @property
    def tokens(self) -> list[str]:
        return tokenize(self.raw)

    @property
    def key(self) -> str:
        return self.tokens[0] if self.raw else ""

    def __repr__(self) -> str:
        return f"Line({self.raw!r})"


class Node:
    def __init__(self, header: str, children: list["Node | Line"] | None = None):
        self.header = header            # raw text after '<'
        self.children: list[Node | Line] = children or []

    @classmethod
    def of(cls, *tokens) -> "Node":
        return cls(format_line(*tokens))

    @property
    def name(self) -> str:
        return tokenize(self.header)[0]

    @property
    def args(self) -> list[str]:
        return tokenize(self.header)[1:]

    def __repr__(self) -> str:
        return f"Node({self.header!r}, {len(self.children)} children)"

    # --- navigation -------------------------------------------------------------------------
    def nodes(self, name: str | None = None) -> Iterator["Node"]:
        return (c for c in self.children if isinstance(c, Node) and (name is None or c.name == name))

    def find(self, name: str) -> "Node | None":
        return next(self.nodes(name), None)

    def line(self, key: str) -> Line | None:
        return next((c for c in self.children if isinstance(c, Line) and c.key == key), None)

    def value(self, key: str) -> list[str] | None:
        ln = self.line(key)
        return ln.tokens[1:] if ln else None

    # --- editing ----------------------------------------------------------------------------
    def set(self, key: str, *values) -> None:
        """Replace the first ``key`` line, or insert it after the last plain line."""
        new = Line.of(key, *values)
        for i, c in enumerate(self.children):
            if isinstance(c, Line) and c.key == key:
                self.children[i] = new
                return
        self.insert_line(new)

    def insert_line(self, line: Line) -> None:
        idx = next((i for i, c in enumerate(self.children) if isinstance(c, Node)), len(self.children))
        self.children.insert(idx, line)

    def remove(self, pred) -> None:
        self.children = [c for c in self.children if not pred(c)]

    def remove_nodes(self, name: str) -> None:
        self.remove(lambda c: isinstance(c, Node) and c.name == name)

    def remove_lines(self, key: str) -> None:
        self.remove(lambda c: isinstance(c, Line) and c.key == key)

    def append(self, child: "Node | Line") -> "Node | Line":
        self.children.append(child)
        return child

    # --- serialisation ----------------------------------------------------------------------
    def dumps(self, indent: int = 0) -> str:
        pad = "  " * indent
        out = [f"{pad}<{self.header}"]
        for c in self.children:
            out.append(c.dumps(indent + 1) if isinstance(c, Node) else f"{pad}  {c.raw}")
        out.append(f"{pad}>")
        return "\n".join(out)


def parse(text: str) -> Node:
    root: Node | None = None
    stack: list[Node] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("<"):
            node = Node(s[1:])
            if stack:
                stack[-1].children.append(node)
            elif root is None:
                root = node
            stack.append(node)
        elif s == ">":
            if not stack:
                raise ValueError("unbalanced '>' in RPP")
            stack.pop()
        elif stack:
            stack[-1].children.append(Line(s))
    if root is None or stack:
        raise ValueError("not a complete RPP document")
    return root


def dumps(root: Node) -> str:
    return root.dumps() + "\n"
