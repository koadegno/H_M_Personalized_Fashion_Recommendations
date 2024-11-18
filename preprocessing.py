from typing import Tuple
import pandas as pd
import numpy as np


# ARTICLES Preprocessing functions


def preprocess_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing articles dataframe. Removes cols with nulls, set the correct type for article_id

    Args:
        df (pd.DataFrame): The dataframe to preprocess

    Returns:
        pd.DataFrame: The preprocessed articles dataframe
    """
    df.dropna(axis=1, inplace=True)
    df["article_id"] = df["article_id"].astype(str)
    # df["prod_name_length"] = create_prod_name_length(df)
    # df["detail_desc_length"] = create_detail_desc_length(df)
    return df


# CUSTOMERS Preprocessing functions


def preprocess_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing customer dataframe.
    - Fill nulls for club members
    - Create Bin for ages
    - Drop null cols

    Args:
        df (pd.DataFrame): The dataframe to preprocess

    Returns:
        pd.DataFrame: The preprocessed customers dataframe
    """
    df["club_member_status"].fillna("Missing", inplace=True)  # Filling missing fields with the value "Missing"
    df.dropna(subset=["age"], inplace=True)

    # Creating age groups using pandas cut function
    age_bins = [0, 18, 25, 35, 45, 55, 65, 80, 100]
    age_labels = ["0-18", "19-25", "26-35", "36-45", "46-55", "56-65", "66-80", "80+"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)

    df.dropna(axis=1, inplace=True)
    return df


# Transaction Preprocessing functions


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing transaction dataframe.
    - Create new columns for the year, month, day
    - Set the correct type for article_id
    - Create Bin for ages
    - Drop null

    Args:
        df (pd.DataFrame): The dataframe to preprocess

    Returns:
        pd.DataFrame: The preprocessed customers dataframe
    """
    """
    Prepares the input DataFrame by applying various transformations on each column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Processed DataFrame with new features.
    """
    df["article_id"] = df["article_id"].astype(str)

    df["t_dat"] = pd.to_datetime(df["t_dat"])
    df["year"] = df["t_dat"].dt.year
    df["month"] = df["t_dat"].dt.month
    df["day"] = df["t_dat"].dt.day
    df["day_of_week"] = df["t_dat"].dt.day_of_week

    # Convert 't_dat' to epoch milliseconds
    df["month_sin"] = df["month"].apply(lambda x: month_to_sincos(x)[0])
    df["month_cos"] = df["month"].apply(lambda x: month_to_sincos(x)[1])

    df["t_dat"].astype(np.int64) / 10**9

    return df


def month_to_sincos(month: pd.Series) -> Tuple[float, float]:
    """Calculates 'month_sin' and 'month_cos' columns based on the 'month' column.
    Source Idea: https://stats.stackexchange.com/questions/311494/best-practice-for-encoding-datetime-in-machine-learning

    Args:
        month (pd.Series): The month column

    Returns:
        Tuple[float,float]: The month sine and the month cosine
    """
    angle = 2 * np.pi * (month - 1) / 12
    return np.sin(angle), np.cos(angle)
