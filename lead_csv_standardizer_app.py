import io
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml


APP_VERSION = "V2"
APP_LAST_UPDATED = "2026-02-27"


US_STATE_ABBREVIATIONS = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
}

US_STATE_NAMES = {
    "ALABAMA",
    "ALASKA",
    "ARIZONA",
    "ARKANSAS",
    "CALIFORNIA",
    "COLORADO",
    "CONNECTICUT",
    "DELAWARE",
    "FLORIDA",
    "GEORGIA",
    "HAWAII",
    "IDAHO",
    "ILLINOIS",
    "INDIANA",
    "IOWA",
    "KANSAS",
    "KENTUCKY",
    "LOUISIANA",
    "MAINE",
    "MARYLAND",
    "MASSACHUSETTS",
    "MICHIGAN",
    "MINNESOTA",
    "MISSISSIPPI",
    "MISSOURI",
    "MONTANA",
    "NEBRASKA",
    "NEVADA",
    "NEW HAMPSHIRE",
    "NEW JERSEY",
    "NEW MEXICO",
    "NEW YORK",
    "NORTH CAROLINA",
    "NORTH DAKOTA",
    "OHIO",
    "OKLAHOMA",
    "OREGON",
    "PENNSYLVANIA",
    "RHODE ISLAND",
    "SOUTH CAROLINA",
    "SOUTH DAKOTA",
    "TENNESSEE",
    "TEXAS",
    "UTAH",
    "VERMONT",
    "VIRGINIA",
    "WASHINGTON",
    "WEST VIRGINIA",
    "WISCONSIN",
    "WYOMING",
}


US_COUNTRY_KEYWORDS = {
    "united states",
    "united states of america",
    "usa",
    "u.s.a.",
    "u.s.",
    "us",
}


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


def clean_phone_number(value: str) -> str:
    """Keep only digits so phone numbers are standardized."""
    if value is None:
        return ""
    text = str(value)
    digits = re.sub(r"\D+", "", text)
    return digits


def map_employee_count_to_range(value: str) -> str:
    """
    Map a numeric employee count into CRM picklist buckets:
    0-95, 0-99, 100-499, 500-999, 1000-1999, 2,000+.
    """
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    # Try to extract the first integer found in the text
    match = re.search(r"\d+", text.replace(",", ""))
    if not match:
        return text  # leave as-is if we can't parse a number

    try:
        count = int(match.group(0))
    except ValueError:
        return text

    if count <= 95:
        return "0-95"
    if count <= 99:
        return "0-99"
    if count <= 499:
        return "100-499"
    if count <= 999:
        return "500-999"
    if count <= 1999:
        return "1000-1999"
    return "2,000+"


def split_us_and_non_us(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the standardized dataframe into US and non-US rows.
    Uses State and any Country / raw_Country columns when present.
    """

    def is_us_row(row: pd.Series) -> bool:
        state = str(row.get("State", "") or "").strip()
        country = ""
        for col in ("Country", "raw_Country", "raw_country"):
            if col in row.index:
                country = str(row.get(col, "") or "").strip()
                if country:
                    break

        state_norm = state.upper()
        country_norm = country.lower()

        if country_norm:
            if any(keyword in country_norm for keyword in US_COUNTRY_KEYWORDS):
                return True
            # Explicit non-US country
            return False

        # Fall back to state if no country information
        if state_norm and (
            state_norm in US_STATE_ABBREVIATIONS or state_norm in US_STATE_NAMES
        ):
            return True

        # If we have some state but it's not a US abbreviation, assume non-US
        if state_norm:
            return False

        # No clear location info: default to US to avoid dropping too many
        return True

    mask_us = df.apply(is_us_row, axis=1)
    us_df = df[mask_us].reset_index(drop=True)
    non_us_df = df[~mask_us].reset_index(drop=True)
    return us_df, non_us_df


def detect_issues(us_df: pd.DataFrame, non_us_df: pd.DataFrame) -> List[str]:
    """Generate human-readable descriptions of suspected data issues."""
    issues: List[str] = []

    if us_df is None and non_us_df is None:
        return issues

    frames = []
    if us_df is not None:
        frames.append(us_df)
    if non_us_df is not None:
        frames.append(non_us_df)

    if not frames:
        return ["No rows were found in the uploaded file."]

    all_df = pd.concat(frames, ignore_index=True)
    total_rows = len(all_df)

    # Missing emails
    if "Email" in all_df.columns:
        missing_email = all_df["Email"].astype(str).str.strip() == ""
        count_missing_email = int(missing_email.sum())
        if count_missing_email > 0:
            issues.append(
                f"{count_missing_email} row(s) are missing an Email value. "
                "These leads may not be usable for email outreach."
            )

    # Suspicious phone numbers (too short or too long after cleaning)
    if "Phone Number" in all_df.columns:
        phone_series = all_df["Phone Number"].astype(str).str.strip()
        non_empty = phone_series != ""
        lengths = phone_series.str.len()
        bad_phones = non_empty & ((lengths < 10) | (lengths > 15))
        count_bad_phones = int(bad_phones.sum())
        if count_bad_phones > 0:
            issues.append(
                f"{count_bad_phones} phone number(s) look suspicious in length "
                "(less than 10 digits or more than 15 digits) after cleaning."
            )

    # Employee headcount values that didn't map cleanly to a known bucket
    allowed_buckets = {"", "0-95", "0-99", "100-499", "500-999", "1000-1999", "2,000+"}
    if "Employee headcount size" in all_df.columns:
        emp_series = all_df["Employee headcount size"].astype(str).str.strip()
        invalid_mask = ~emp_series.isin(allowed_buckets)
        invalid_counts = emp_series[invalid_mask].value_counts().head(5)
        invalid_total = int(invalid_mask.sum())
        if invalid_total > 0:
            examples = ", ".join(list(invalid_counts.index))
            issues.append(
                f"{invalid_total} row(s) have employee headcount values that did not "
                f"map cleanly into CRM buckets. Example values: {examples}."
            )

    # Non-US leads present
    if non_us_df is not None and not non_us_df.empty:
        count_non_us = len(non_us_df)
        issues.append(
            f"{count_non_us} lead(s) were classified as non-US and moved to the "
            "separate non-US CSV. Review them if you expected only US leads."
        )

    # Unmapped/raw columns
    raw_cols = [c for c in all_df.columns if c.startswith("raw_")]
    if raw_cols:
        issues.append(
            f"{len(raw_cols)} column(s) were not mapped into the standard template "
            "and are kept as raw_ columns. You may want to update the template "
            "or handle these manually."
        )

    if not issues:
        issues.append("No obvious data issues detected.")

    return issues


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

    # Add blank call-related columns that will be filled in later
    call_columns = ["Assigned To", "Tier", "Member Status"]
    for col in call_columns:
        standardized[col] = ""

    if keep_unmapped:
        used_vendor_cols = {v for v in mapping.values() if v is not None}
        for col in df.columns:
            if col not in used_vendor_cols:
                standardized[f"{unmapped_prefix}{col}"] = df[col]

    # Remove duplicate columns and enforce column order:
    # standard fields -> call columns -> unmapped "raw_" columns to the far right.
    standardized = standardized.loc[:, ~standardized.columns.duplicated()]
    raw_cols = [
        c
        for c in standardized.columns
        if c not in standard_columns and c not in call_columns
    ]
    ordered_cols = standard_columns + call_columns + raw_cols
    standardized = standardized[ordered_cols]
    return standardized


def main() -> None:
    st.set_page_config(
        page_title="Lead CSV Standardizer",
        page_icon="📊",
        layout="wide",
    )

    st.title("Lead CSV Standardizer")
    st.caption(f"Version {APP_VERSION} · Last updated {APP_LAST_UPDATED}")
    st.write(
        "Upload any vendor lead list CSV and map it into a single, "
        "standard structure your whole team can use."
    )

    # Required user info for auditing
    st.markdown("### User details")
    user_name = st.text_input("Your name (required)")
    user_email = st.text_input("Your email (required)")

    # Initialize session state for generated outputs and issues
    if "us_df" not in st.session_state:
        st.session_state["us_df"] = None
    if "non_us_df" not in st.session_state:
        st.session_state["non_us_df"] = None
    if "issues" not in st.session_state:
        st.session_state["issues"] = []

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

    generate_disabled = not (user_name.strip() and user_email.strip())

    if generate_disabled:
        st.info("Enter your name and email above to enable generation.")

    if st.button("Generate standardized CSV", type="primary", disabled=generate_disabled):
        standardized_df = apply_mapping_to_df(df, config, editable_mapping)

        # Standardize phone numbers
        if "Phone Number" in standardized_df.columns:
            standardized_df["Phone Number"] = (
                standardized_df["Phone Number"]
                .astype(str)
                .fillna("")
                .map(clean_phone_number)
            )

        # Map employee headcount size into CRM picklist ranges
        if "Employee headcount size" in standardized_df.columns:
            standardized_df["Employee headcount size"] = (
                standardized_df["Employee headcount size"]
                .astype(str)
                .fillna("")
                .map(map_employee_count_to_range)
            )

        # Split into US and non-US based on location
        us_df, non_us_df = split_us_and_non_us(standardized_df)

        st.session_state["us_df"] = us_df
        st.session_state["non_us_df"] = non_us_df

        # Analyze suspected issues
        issues = detect_issues(us_df, non_us_df)
        st.session_state["issues"] = issues

        # Log usage to a CSV file
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_name": user_name.strip(),
            "user_email": user_email.strip(),
            "total_rows": len(standardized_df),
            "us_rows": len(us_df),
            "non_us_rows": len(non_us_df),
            "issue_count": len(issues),
            "issues_summary": "; ".join(issues),
        }
        log_path = "usage_log.csv"
        try:
            if os.path.exists(log_path):
                log_df = pd.read_csv(log_path)
                log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
            else:
                log_df = pd.DataFrame([log_entry])
            log_df.to_csv(log_path, index=False)
        except Exception:
            # Logging failures should not break the app
            pass

        st.success(
            f"Standardized data generated. US rows: {len(us_df)}, "
            f"non-US rows: {len(non_us_df)}."
        )

    us_df = st.session_state.get("us_df")
    non_us_df = st.session_state.get("non_us_df")
    issues = st.session_state.get("issues", [])

    st.subheader("Suspected issues in this file")
    if issues:
        for msg in issues:
            st.write(f"- {msg}")
    else:
        st.write("No obvious data issues detected yet. Generate a file to see results.")

    if us_df is not None:
        st.markdown("**Preview · US leads (first 50 rows)**")
        st.dataframe(us_df.head(50))

        csv_buffer_us = io.StringIO()
        us_df.to_csv(csv_buffer_us, index=False)
        csv_bytes_us = csv_buffer_us.getvalue().encode("utf-8")

        st.download_button(
            label="Download standardized US CSV",
            data=csv_bytes_us,
            file_name="standardized_leads_us.csv",
            mime="text/csv",
        )

    if non_us_df is not None and not non_us_df.empty:
        st.markdown("**Preview · Non-US leads (first 50 rows)**")
        st.dataframe(non_us_df.head(50))

        csv_buffer_non_us = io.StringIO()
        non_us_df.to_csv(csv_buffer_non_us, index=False)
        csv_bytes_non_us = csv_buffer_non_us.getvalue().encode("utf-8")

        st.download_button(
            label="Download non-US leads CSV",
            data=csv_bytes_non_us,
            file_name="standardized_leads_non_us.csv",
            mime="text/csv",
        )

    st.subheader("Usage log")
    try:
        if os.path.exists("usage_log.csv"):
            log_df = pd.read_csv("usage_log.csv")
            st.dataframe(log_df.tail(50))
        else:
            st.write("No usage has been logged yet.")
    except Exception as e:  # noqa: BLE001
        st.write(f"Could not read usage log: {e}")


if __name__ == "__main__":
    main()

