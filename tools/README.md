# Repository tools

`tools` contains small repository-level utilities that protect the published
source boundary. They are intentionally separate from Arena, ADK, the Console,
and the gym so a clean checkout can run publication checks without importing
the training stack.

## Entry points

- [`check_publication_hygiene.py`](check_publication_hygiene.py) asserts that
  private, generated, and third-party material remains excluded while required
  source remains trackable. Run it on the published tree: since 2026-10-25 the
  owner's local repository also tracks private trees that are never published.

For runtime and training utilities, continue to the package that owns the
corresponding interface rather than adding them here.
