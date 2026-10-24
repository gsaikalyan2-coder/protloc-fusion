"""Generate ``frontend/src/styles/tokens.css`` from ``docs/DESIGN.md``.

A design system that lives in a document and a stylesheet that lives in code
drift apart within a fortnight, and the drift is invisible because both look
right on their own. This script removes the second copy: ``docs/DESIGN.md`` is
the source, ``tokens.css`` is derived, and CI regenerates it and fails on a diff.

Two kinds of source are read.

*The token blocks* in §1 and §3.3 are already CSS. Their ``:root`` rules are
parsed and re-emitted in document order, so adding a token to the document is
all that adding a token takes.

*The compartment tables* in §2.1 and §2.2 are Markdown. Each row carries a
token name and four colours — light fill, light ink, dark fill, dark ink — which
become ``--loc-<code>-fill`` and ``--loc-<code>-ink`` in the light and dark
blocks. A row whose colours read *(as above)* is a cross-reference to a
compartment already defined and contributes nothing.

Run it from the repository root::

    python scripts/generate_tokens.py          # write the file
    python scripts/generate_tokens.py --check  # exit 1 if the file is stale
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
DESIGN_DOC = REPO_ROOT / "docs" / "DESIGN.md"
OUTPUT = REPO_ROOT / "frontend" / "src" / "styles" / "tokens.css"

LIGHT_SELECTOR = ":root"
DARK_SELECTOR = ':root[data-theme="dark"]'

# A ```css fenced block.
_CSS_FENCE = re.compile(r"```css\n(.*?)```", re.DOTALL)
# A `selector { ... }` rule inside such a block.
_CSS_RULE = re.compile(r"([^{}]+?)\{([^{}]*)\}", re.DOTALL)
# A `--name: value;` declaration. Comments are removed before this runs.
_DECLARATION = re.compile(r"(--[a-z0-9-]+)\s*:\s*([^;]+);")
# A CSS comment, which may span lines and may itself mention a token name.
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
# A Markdown table row whose fourth cell is a `--loc-*` token in backticks.
_HEX = r"#[0-9A-Fa-f]{6}"
_PALETTE_ROW = re.compile(
    r"^\|\s*\d+\s*\|\s*(?P<code>[A-Z]+)\s*\|[^|]*\|\s*`(?P<token>--loc-[a-z]+)`\s*\|"
    r"\s*`(?P<light_fill>" + _HEX + r")`\s*\|"
    r"\s*`(?P<light_ink>" + _HEX + r")`\s*\|"
    r"\s*`(?P<dark_fill>" + _HEX + r")`\s*\|"
    r"\s*`(?P<dark_ink>" + _HEX + r")`\s*\|",
    re.MULTILINE,
)

HEADER = """/* ------------------------------------------------------------------------
 * GENERATED FILE — DO NOT EDIT.
 *
 * Produced by scripts/generate_tokens.py from docs/DESIGN.md.
 * Change a colour, a size or a compartment in docs/DESIGN.md and re-run:
 *
 *     python scripts/generate_tokens.py
 *
 * CI regenerates this file and fails the build if the result differs from
 * what is committed, so an edit made here is reverted rather than kept.
 * ------------------------------------------------------------------------ */
"""


class TokenGenerationError(RuntimeError):
    """Raised when ``docs/DESIGN.md`` cannot produce a complete token set."""


def _strip_at_rules(css: str) -> str:
    """Remove every ``@media``/``@keyframes``/... block, nested rules included.

    The flat rule pattern cannot see nesting, so without this the ``:root`` rule
    inside ``@media (prefers-reduced-motion: reduce)`` was read as a plain
    ``:root`` rule and overwrote the base motion tokens with ``0ms``. At-rules
    are emitted from the template in :func:`render_tokens` instead.
    """
    out: list[str] = []
    i = 0
    while i < len(css):
        if css[i] == "@":
            brace = css.find("{", i)
            if brace == -1:
                break
            depth = 0
            j = brace
            while j < len(css):
                if css[j] == "{":
                    depth += 1
                elif css[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            i = j + 1
            continue
        out.append(css[i])
        i += 1
    return "".join(out)


def _parse_css_blocks(markdown: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return the light and dark declarations from the document's CSS blocks.

    Declarations are keyed by custom-property name, so a later block redefining
    a token overrides an earlier one exactly as the cascade would.
    """
    light: dict[str, str] = {}
    dark: dict[str, str] = {}

    for block in _CSS_FENCE.findall(markdown):
        # Motion and colour-scheme at-rules are emitted from the template in
        # render_tokens rather than copied, so the dark block is written once
        # and guarded consistently.
        flat = _strip_at_rules(_COMMENT.sub("", block))
        for raw_selector, body in _CSS_RULE.findall(flat):
            selector = " ".join(raw_selector.split())
            declarations = {
                name: " ".join(value.split()) for name, value in _DECLARATION.findall(body)
            }
            if not declarations:
                continue
            if selector == DARK_SELECTOR:
                dark.update(declarations)
            elif selector == LIGHT_SELECTOR:
                light.update(declarations)

    return light, dark


def _parse_palette(markdown: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return the compartment fill and ink tokens for both themes."""
    light: dict[str, str] = {}
    dark: dict[str, str] = {}

    for match in _PALETTE_ROW.finditer(markdown):
        token = match.group("token")
        light[f"{token}-fill"] = match.group("light_fill")
        light[f"{token}-ink"] = match.group("light_ink")
        dark[f"{token}-fill"] = match.group("dark_fill")
        dark[f"{token}-ink"] = match.group("dark_ink")

    return light, dark


def _render_block(selector: str, declarations: dict[str, str], indent: str = "  ") -> str:
    """Render one CSS rule with its declarations aligned on the value column."""
    if not declarations:
        return ""
    width = max(len(name) for name in declarations) + 1
    lines = "\n".join(
        f"{indent}{name + ':':<{width + 1}} {value};" for name, value in declarations.items()
    )
    return f"{selector} {{\n{lines}\n}}\n"


def _indent(block: str, wrapper: str) -> str:
    """Nest a rendered CSS rule inside an at-rule, indenting it by one level."""
    body = "\n".join(f"  {line}" if line else line for line in block.rstrip("\n").split("\n"))
    return f"{wrapper} {{\n{body}\n}}\n"


def render_tokens(markdown: str) -> str:
    """Build the whole stylesheet from the document's text."""
    css_light, css_dark = _parse_css_blocks(markdown)
    palette_light, palette_dark = _parse_palette(markdown)

    if not css_light:
        message = "no :root token block found in docs/DESIGN.md §1"
        raise TokenGenerationError(message)
    if not palette_light:
        message = "no compartment palette rows found in docs/DESIGN.md §2"
        raise TokenGenerationError(message)

    light = {**css_light, **palette_light}
    dark = {**css_dark, **palette_dark}

    missing = sorted(name for name in dark if name not in light)
    if missing:
        message = f"tokens defined only for the dark theme: {missing}"
        raise TokenGenerationError(message)

    parts = [
        HEADER,
        "\n/* Light theme — the default. */\n",
        _render_block(LIGHT_SELECTOR, light),
        "\n/* Dark theme — an explicit choice by the viewer. */\n",
        _render_block(DARK_SELECTOR, dark),
        "\n/* Dark theme — the viewer expressed no preference and the system is dark.\n"
        "   Guarded so that an explicit light choice still wins. */\n",
        _indent(
            _render_block(':root:not([data-theme="light"])', dark),
            "@media (prefers-color-scheme: dark)",
        ),
        "\n/* Motion is removed, never merely shortened, for viewers who ask for that. */\n",
        "@media (prefers-reduced-motion: reduce) {\n"
        "  :root {\n"
        "    --duration-fast: 0ms;\n"
        "    --duration-base: 0ms;\n"
        "  }\n"
        "}\n",
    ]
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Write ``tokens.css``, or check that the committed copy is current.

    Returns:
        ``0`` on success, ``1`` when ``--check`` finds the file stale or the
        document cannot produce a complete token set.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the committed file differs from the generated one",
    )
    args = parser.parse_args(argv)

    if not DESIGN_DOC.is_file():
        sys.stderr.write(f"design document not found: {DESIGN_DOC}\n")
        return 1

    try:
        rendered = render_tokens(DESIGN_DOC.read_text(encoding="utf-8"))
    except TokenGenerationError as exc:
        sys.stderr.write(f"cannot generate tokens: {exc}\n")
        return 1

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        if current != rendered:
            sys.stderr.write(
                f"{OUTPUT.relative_to(REPO_ROOT)} is out of date with docs/DESIGN.md.\n"
                "Run: python scripts/generate_tokens.py\n"
            )
            return 1
        sys.stdout.write(f"{OUTPUT.relative_to(REPO_ROOT)} is current\n")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8", newline="\n")
    token_count = rendered.count("--") // 2
    sys.stdout.write(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({token_count} declarations)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
