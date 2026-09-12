# UI Language

The application uses one language selector in the sidebar for both LLM presentation output and the visible application UI.

- `English`: English UI labels and English presentation output.
- `한국어`: Korean UI labels and Korean presentation output.

The language selector does not change database values, workflow keys, IDs, ontology identifiers, paper titles, or stored research records. This keeps both languages on one codebase and the same workspace databases.

The translation layer is implemented with `ui_text(ko, en)` in `app.py`. New user-facing UI copy should use this helper rather than creating a separate English app.
