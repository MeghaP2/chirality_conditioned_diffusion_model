"""Console script for chirality_aware_conformer_generation."""

import typer
from rich.console import Console

from chirality_aware_conformer_generation import utils

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
    """Console script for chirality_aware_conformer_generation."""
    console.print("Replace this message by putting your code into chirality_aware_conformer_generation.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
