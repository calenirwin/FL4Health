from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fl4health.feature_alignment.constants import TextFeatureTransformer


class TextMulticolumnTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, transformer: TextFeatureTransformer):
        """
        The purpose of this class is to enable the application of text feature transformers from sklearn
        to multiple string columns, which is not supported in the first place.

        Args:
            transformer (TextFeatureTransformer): Transformer to be applied
        """
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> TextMulticolumnTransformer:
        """
        Fit the transformer to the provided dataframe. The dataframe should have multiple string columns
        The transformer is fit on the appended text from all columns in the ``X`` dataframe

        Args:
            X (pd.DataFrame): Columns on which to fit the transformer
            y (pd.DataFrame | None, optional): Not used. Defaults to None.

        Returns:
            TextMulticolumnTransformer: The fit transformer
        """
        joined_X = X.apply(lambda x: " ".join(x), axis=1)
        self.transformer.fit(joined_X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the concatenation of all columns of text in the ``X`` dataframe

        Args:
            X (pd.DataFrame): Dataframe of text-based columns to be transformed

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        joined_X = X.apply(lambda x: " ".join(x), axis=1)
        return self.transformer.transform(joined_X)


class TextColumnTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, transformer: TextFeatureTransformer):
        """
        The purpose of this class is to enable the application of text feature transformers from sklearn
        to a single-column pandas dataframe, which is not supported in the first place.

        Args:
            transformer (TextFeatureTransformer): Transformer to be applied
        """
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> TextColumnTransformer:
        """
        Fit the transformer to the provided dataframe. The dataframe should have a single string column
        The transformer is fit on the text from the single columns in the ``X`` dataframe

        Args:
            X (pd.DataFrame): Column on which to fit the transformer
            y (pd.DataFrame | None, optional): Not used. Defaults to None.

        Returns:
            TextColumnTransformer: The fit transformer
        """
        assert isinstance(X, pd.DataFrame) and X.shape[1] == 1
        self.transformer.fit(X[X.columns[0]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the concatenation of a single column of text in the ``X`` dataframe

        Args:
            X (pd.DataFrame): Dataframe of text-based column to be transformed

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        assert isinstance(X, pd.DataFrame) and X.shape[1] == 1
        return self.transformer.transform(X[X.columns[0]])
