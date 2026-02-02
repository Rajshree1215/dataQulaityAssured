"""
Custom date validation functions
"""
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger("DQPipeline.date_validators")


def validate_date_of_birth(
    df: pd.DataFrame,
    column: str,
    rules: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate Date of Birth column
    
    Rules:
    - Format: DD/MM/YYYY
    - Person must be > min_age years old
    - Year must be >= min_year (e.g., 1900)
    - Year must be <= max_year (e.g., current year)
    
    Args:
        df: DataFrame
        column: Column name (e.g., "Date of Birth")
        rules: Validation rules from schema
    
    Returns:
        Validation result dictionary
    """
    date_format = rules.get("date_format", "DD/MM/YYYY")
    min_age = rules.get("min_age", 18)
    min_year = rules.get("min_year", 1900)
    max_year = rules.get("max_year", datetime.now().year)
    
    current_year = datetime.now().year
    max_birth_year = current_year - min_age  # e.g., 2026 - 18 = 2008
    
    failures = []
    
    # Get the column
    dob_series = df[column]
    
    # Check each value
    for idx, dob_value in dob_series.items():
        # Skip nulls (handled by nullable_pct validation)
        if pd.isnull(dob_value):
            continue
        
        try:
            # Parse date (DD/MM/YYYY format)
            dob_str = str(dob_value).strip()
            dob = datetime.strptime(dob_str, "%d/%m/%Y")
            
            birth_year = dob.year
            birth_date = dob.date()
            current_date = datetime.now().date()
            
            # Calculate age
            age = current_year - birth_year
            # Adjust if birthday hasn't occurred this year
            if (current_date.month, current_date.day) < (dob.month, dob.day):
                age -= 1
            
            # Validation 1: Age must be >= min_age
            if age < min_age:
                failures.append({
                    "row": idx,
                    "value": dob_str,
                    "age": age,
                    "reason": f"Person is {age} years old, must be >= {min_age}"
                })
                continue
            
            # Validation 2: Birth year must be >= min_year
            if birth_year < min_year:
                failures.append({
                    "row": idx,
                    "value": dob_str,
                    "birth_year": birth_year,
                    "reason": f"Birth year {birth_year} is before {min_year}"
                })
                continue
            
            # Validation 3: Birth date cannot be in the future
            if birth_date > current_date:
                failures.append({
                    "row": idx,
                    "value": dob_str,
                    "reason": f"Birth date {dob_str} is in the future"
                })
                continue
            
            # Validation 4: Birth year cannot be > max_year
            if birth_year > max_year:
                failures.append({
                    "row": idx,
                    "value": dob_str,
                    "birth_year": birth_year,
                    "reason": f"Birth year {birth_year} is after {max_year}"
                })
                continue
                
        except ValueError as e:
            # Invalid date format
            failures.append({
                "row": idx,
                "value": dob_value,
                "reason": f"Invalid date format (expected DD/MM/YYYY): {str(e)}"
            })
    
    # Prepare result
    total_rows = len(dob_series.dropna())
    failed_count = len(failures)
    success = failed_count == 0
    
    result = {
        "success": success,
        "column": column,
        "expectation_type": "custom_date_of_birth_validation",
        "result": {
            "element_count": total_rows,
            "unexpected_count": failed_count,
            "unexpected_percent": round((failed_count / total_rows * 100), 2) if total_rows > 0 else 0,
            "unexpected_list": [f["value"] for f in failures[:20]],  # First 20
            "failure_details": failures[:20]  # First 20 with reasons
        }
    }
    
    if not success:
        logger.warning(
            f"Date of Birth validation failed: {failed_count} invalid dates "
            f"({result['result']['unexpected_percent']}%)"
        )
    
    return result


def validate_date_format(
    value: str,
    date_format: str = "DD/MM/YYYY"
) -> bool:
    """
    Check if a date string matches expected format
    
    Args:
        value: Date string
        date_format: Expected format
    
    Returns:
        True if valid, False otherwise
    """
    try:
        if date_format == "DD/MM/YYYY":
            datetime.strptime(value, "%d/%m/%Y")
        elif date_format == "MM/DD/YYYY":
            datetime.strptime(value, "%m/%d/%Y")
        elif date_format == "YYYY-MM-DD":
            datetime.strptime(value, "%Y-%m-%d")
        else:
            return False
        return True
    except (ValueError, TypeError):
        return False