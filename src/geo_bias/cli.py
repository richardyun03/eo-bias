"""Top-level CLI entry point referenced by [project.scripts] in pyproject.toml.

Subcommands will be wired up as the numbered scripts under scripts/ stabilize.
"""

import click


@click.group()
def main() -> None:
    """geo-bias: cross-region benchmark for EO foundation models."""


if __name__ == "__main__":
    main()
