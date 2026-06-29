import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# category_encoders is needed for Target Encoding
try:
    # pyrefly: ignore [missing-import]
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders not installed. Target Encoding will be skipped.")

def main():
    print("Loading Dataset...")
    file_path = "chd.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # chd.csv is tab-separated
    df = pd.read_csv(file_path, sep='\t')
    print(
        f"Dataset loaded successfully.\n"
        f"Rows: {df.shape[0]}\n"
        f"Columns: {df.shape[1]}"
    )
    print("\nColumn names:")
    print(df.columns.tolist())

    # Handling Missing Values demonstration
    print("\nHandling Missing DATA...")
    print("Artificially setting some 'MedInc' values to NaN for demonstration...")

    # Artificially create missing values in MedInc
    df.loc[0:24, 'MedInc'] = np.nan

    # Impute missing values with the median
    imputer = SimpleImputer(strategy='median')
    df[['MedInc']] = imputer.fit_transform(df[['MedInc']])
    print(
        f"Imputation complete. "
        f"'MedInc' column now has {df['MedInc'].isnull().sum()} missing values."
    )
    print("\nFirst 30 rows of MedInc:")
    print(df['MedInc'].head(30))

    # High cardinality — create a synthetic categorical column to demonstrate Target Encoding
    df['Region_ID'] = [f"Region_{np.random.randint(1, 50)}" for _ in range(len(df))]

    # Target variable: we will predict MedInc (median income)
    # so for encoding, use MedInc as the target
    if TargetEncoder is not None:
        print("\nApplying Target Encoder on Region_ID...")
        encoder = TargetEncoder()
        df['Region_ID_Encoded'] = encoder.fit_transform(df['Region_ID'], df['MedInc'])
        print("Target encoding complete. First 5 rows:")
        print(df[['Region_ID', 'MedInc', 'Region_ID_Encoded']].head())
    else:
        print("\nCategory Encoder not installed")

    # Feature selection — use housing features to predict MedInc
    features_to_test = ['HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    x_features = df[features_to_test].fillna(0)
    y_target = df['MedInc']

    selector = SelectKBest(score_func=mutual_info_regression, k=2)
    selector.fit(x_features, y_target)

    winning_features = selector.get_support()
    best_features = x_features.columns[winning_features].tolist()

    print("\nSelected features:")
    print(best_features)

    # Splitting data
    x = df[best_features]
    y = df['MedInc']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"Training data size: {x_train.shape}")
    print(f"Testing data size:  {x_test.shape}\n")

    # Training model
    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    # Show first 3 predictions vs actual
    actual_values = y_test.head(3).values
    predicted_values = predictions[:3]

    print("Sample Predictions vs Actual:")
    for i in range(3):
        predicted = round(predicted_values[i], 2)
        actual = round(actual_values[i], 2)
        difference = round(abs(actual - predicted), 2)

        print(f"  Model Guessed : {predicted}")
        print(f"  Real Answer   : {actual}")
        print(f"  Difference    : {difference}")
        print()

if __name__ == '__main__':
    main()