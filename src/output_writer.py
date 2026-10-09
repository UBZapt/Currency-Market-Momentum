"""Consolidated workbook writer for §3 onward results tables."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
WORKBOOK_PATH = PROJECT_ROOT / "output" / "results.xlsx"

_sheets: "dict[str, pd.DataFrame]" = {}


def add_sheet(name: str, df: pd.DataFrame) -> None:
    """Register a DataFrame as a sheet in the consolidated results workbook."""
    sheet = name[:31]
    _sheets[sheet] = df
    print(f"Loaded: {sheet}")


def write_workbook() -> None:
    """Flush all registered sheets to output/results.xlsx."""
    if not _sheets:
        return
    WORKBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(WORKBOOK_PATH, engine="openpyxl") as w:
        for name, df in _sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    print(f"Written: {WORKBOOK_PATH.name} ({len(_sheets)} sheets)")


def reset() -> None:
    """Clear the in-memory sheet registry (used by tests/reruns)."""
    _sheets.clear()
