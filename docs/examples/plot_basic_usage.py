"""
=========================
Transformer Demonstration
=========================

This example demonstrates the basic usage of PhilanthroPy
transformers within a scikit-learn pipeline to process donor data.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from philanthropy.preprocessing import WealthScreeningImputer

# Create a sample dataset
X = pd.DataFrame({
    'WEALTH_RATING': [1.0, np.nan, 3.0, np.nan, 5.0],
    'PAST_GIVING': [100, 50, 500, 20, 1000]
})

print("Original Data:")
print(X)

# Initialize the pipeline
pipeline = Pipeline([
    ('imputer', WealthScreeningImputer(strategy='median', add_indicator=True))
])

# Fit and transform
X_transformed = pipeline.fit_transform(X[['WEALTH_RATING']])

print("\nTransformed Wealth Ratings (with indicator on last column):")
print(X_transformed)
