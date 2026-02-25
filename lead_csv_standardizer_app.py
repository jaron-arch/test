import io
import re
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml


def load_config(path: str = "standard_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_header(name: str) -> str:
    """Lowercase, trim, and strip non-alphanumeric characters for comparison."""
    name = name.strip().lower()
    name = re.sub(r"[_\-\s]+", " ", name)
    name = re.sub(r"[^0-9a-z]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def build_synonym_index(config: dict) -> Dict[str, List[str]]:
    """Map normalized synonym -> list of standard field IDs that accept it."""
    index: Dict[str, List[str]] = {}
    for field in config.get("standard_fields", []):
        field_id = field["id"]
        for syn in field.get("synonyms", []):
            key = normalize_header(syn)
            index.setdefault(key, []).append(field_id)
    return index


def auto_map_columns(
    vendor_columns: List[str], config: dict
) -> Dict[str, Optional[str]]:
    """
    Return mapping of standard_field_id -> vendor_column (or None if not found).
    """
    synonym_index = build_synonym_index(config)
    normalized_vendor = {normalize_header(c): c for c in vendor_columns}

    mapping: Dict[str, Optional[str]] = {}
    for field in config.get("standard_fields", []):
        field_id = field["id"]
        candidates = []
        for syn in field.get("synonyms", []):
            key = normalize_header(syn)
            if key in normalized_vendor:
                candidates.append(normalized_vendor[key])

        mapping[field_id] = candidates[0] if candidates else None

    return mapping


def apply_mapping_to_df(
    df: pd.DataFrame,
    config: dict,
    mapping: Dict[str, Optional[str]],
) -> pd.DataFrame:
    df = df.copy()

    standard_columns = [f["id"] for f in config.get("standard_fields", [])]
    options = config.get("options", {})
    keep_unmapped = bool(options.get("keep_unmapped_columns", True))
    unmapped_prefix = str(options.get("unmapped_prefix", "raw_"))

    standardized = pd.DataFrame()

    for field in config.get("standard_fields", []):
        field_id = field["id"]
        vendor_col = mapping.get(field_id)
        if vendor_col and vendor_col in df.columns:
            standardized[field_id] = df[vendor_col]
        else:
            standardized[field_id] = ""

    if keep_unmapped:
        used_vendor_cols = {v for v in mapping.values() if v is not None}
        for col in df.columns:
            if col not in used_vendor_cols:
                standardized[f"{unmapped_prefix}{col}"] = df[col]

    standardized = standardized.loc[:, ~standardized.columns.duplicated()]
    return standardized


def main() -> None:
    st.set_page_config(
        page_title="Lead CSV Standardizer",
        page_icon="📊",
        layout="wide",
    )

    st.title("Lead CSV Standardizer")
    st.write(
        "Upload any vendor lead list CSV and map it into a single, "
        "standard structure your whole team can use."
    )

    try:
        config = load_config()
    except FileNotFoundError:
        st.error(
            "Could not find `standard_config.yaml` in this folder. "
            "Please make sure it exists next to this app file."
        )
        st.stop()

    standard_fields = config.get("standard_fields", [])

    with st.expander("View / edit standard field template", expanded=False):
        st.write(
            "These are the standard fields and synonyms used to auto-detect "
            "columns from vendor CSVs. You can edit them in `standard_config.yaml`."
        )
        st.json(standard_fields)

    uploaded_file = st.file_uploader(
        "Upload a lead list CSV",
        type=["csv"],
        accept_multiple_files=False,
    )

    if not uploaded_file:
        st.info("Upload a CSV file to begin.")
        return

    try:
        data = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(data), dtype=str, encoding="utf-8-sig")
    except Exception as e:  # noqa: BLE001
        st.error(f"Error reading CSV file: {e}")
        return

    df.columns = [str(c) for c in df.columns]

    st.subheader("Step 1 · Review detected columns")
    st.write("These are the column headers found in your uploaded CSV:")
    st.write(list(df.columns))

    st.subheader("Step 2 · Auto-mapping to your standard schema")
    auto_mapping = auto_map_columns(list(df.columns), config)

    st.write(
        "You can adjust any of these mappings. "
        "For each standard field, choose which vendor column should feed it."
    )

    col1, col2 = st.columns([1.5, 2.5])

    with col1:
        st.markdown("**Standard fields & mappings**")

        editable_mapping: Dict[str, Optional[str]] = {}
        for field in standard_fields:
            field_id = field["id"]
            label = field.get("label", field_id)
            required = field.get("required", False)

            vendor_options = ["(none)"] + list(df.columns)
            default_vendor = auto_mapping.get(field_id)
            if default_vendor is None or default_vendor not in vendor_options:
                default_index = 0
            else:
                default_index = vendor_options.index(default_vendor)

            selection = st.selectbox(
                f"{label} ({field_id})" + (" *" if required else ""),
                options=vendor_options,
                index=default_index,
                key=f"mapping_{field_id}",
            )

            editable_mapping[field_id] = None if selection == "(none)" else selection

    with col2:
        st.markdown("**Vendor columns not yet mapped**")

        mapped_vendor_cols = {
            v for v in editable_mapping.values() if v is not None
        }
        unmapped_vendor_cols = [
            c for c in df.columns if c not in mapped_vendor_cols
        ]

        if unmapped_vendor_cols:
            st.write(
                "These columns are not currently mapped to any standard field:"
            )
            st.write(unmapped_vendor_cols)
        else:
            st.write("All vendor columns are mapped to some field.")

    st.subheader("Step 3 · Generate standardized file")

    if st.button("Generate standardized CSV", type="primary"):
        standardized_df = apply_mapping_to_df(df, config, editable_mapping)

        st.success("Standardized data generated.")

        st.markdown("**Preview (first 50 rows)**")
        st.dataframe(standardized_df.head(50))

        csv_buffer = io.StringIO()
        standardized_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="Download standardized CSV",
            data=csv_bytes,
            file_name="standardized_leads.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

