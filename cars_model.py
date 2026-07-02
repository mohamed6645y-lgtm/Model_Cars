
import pandas as pd
import numpy as np
import re
import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import joblib
import json
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

def load_and_clean_data(filepath):
    """Loads data, cleans the price column, and finds price and year columns."""
    df = pd.read_csv(filepath)

    # Create a unique ID from the initial index before any rows are dropped.
    # This ensures the ID is stable and unique to the original row.
    # We check if an 'id' column already exists to avoid overwriting it.
    if 'id' not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)
    
    # Find price column
    price_col = next((col for col in df.columns if "price" in col.lower()), None)
    if not price_col:
        raise ValueError("No 'price' column found in the dataset.")
    
    # Find year column dynamically
    year_col = next((col for col in df.columns if "year" in col.lower()), None)
    if not year_col:
        print("Warning: No 'year' column found. Proceeding without 'car_age' feature.")

    # Drop rows where essential information is missing
    subset_to_drop = [price_col]
    if year_col:
        subset_to_drop.append(year_col)
    df = df.dropna(subset=subset_to_drop)

    def clean_price(x):
        x = str(x)
        x = re.sub(r"[^\d.]", "", x)
        return float(x) if x != "" else np.nan

    df[price_col] = df[price_col].apply(clean_price)
    df = df.dropna(subset=[price_col])

    return df, price_col, year_col

def feature_engineering(df, year_col):
    """Creates new features to improve model performance if year column exists."""
    if year_col:
        current_year = datetime.datetime.now().year
        # Ensure year column is numeric, coercing errors to NaN
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        df = df.dropna(subset=[year_col]) # Drop rows where year could not be converted
        df[year_col] = df[year_col].astype(int)

        df['car_age'] = current_year - df[year_col]
        df = df.drop(year_col, axis=1)
    return df

def build_pipeline(numeric_features, categorical_features):
    """Builds a full preprocessing and modeling pipeline."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    return pipeline

def train_model(X_train, y_train, pipeline):
    """Trains the model using RandomizedSearchCV for hyperparameter tuning."""

    # Define the parameter distribution for RandomizedSearchCV
    param_dist = {
        'regressor__n_estimators': randint(100, 500),
        'regressor__max_depth': [None] + list(np.arange(10, 31, 5)),
        'regressor__min_samples_split': randint(2, 11),
        'regressor__min_samples_leaf': randint(1, 5)
    }

    # Use RandomizedSearchCV to find the best hyperparameters
    # n_iter controls how many different combinations to try. Increase for better results.
    # n_jobs is set to 1 to avoid potential multiprocessing issues on Windows.
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=10, # Reduced n_iter for faster debugging
        cv=3,      # Reduced cv folds for faster debugging
        verbose=1, 
        n_jobs=1, 
        random_state=42
    )
    
    print("Starting model tuning with RandomizedSearchCV...", flush=True)
    search.fit(X_train, y_train)
    
    print(f"Best parameters found: {search.best_params_}", flush=True)
    best_model = search.best_estimator_
    
    return best_model

def evaluate_predictions(model, X_test, y_test, image_map_df=None):
    """Makes predictions and evaluates them based on a percentage difference."""
    # Ensure index is aligned for correct data retrieval
    X_test_copy = X_test.copy()
    y_test_copy = y_test.copy()

    # --- Verification Step ---
    # Check if 'id' column exists in the test set before making predictions.
    if 'id' not in X_test_copy.columns:
        raise KeyError("'id' column not found in X_test. It was likely dropped during preprocessing.")

    print("Evaluating model on the test set...", flush=True)
    score = model.score(X_test, y_test)
    print(f"Model R^2 Score on Test Set: {score:.4f}", flush=True)


    predictions = model.predict(X_test)

    results_df = pd.DataFrame({
        'id': X_test_copy['id'],
        'brand': X_test_copy['brand'],
        'model': X_test_copy['model'],
        'actual_price': y_test_copy,
        'predicted_price': predictions
    })

    def get_evaluation(row):
        # Avoid division by zero
        if row['actual_price'] == 0:
            return "Cannot Evaluate" # Or handle as you see fit
        
        difference = row['actual_price'] - row['predicted_price']
        diff_percent = difference / row['actual_price']
        
        if diff_percent > 0.15:  # Predicted price is >15% lower than actual
            return "Good Deal 🔥"
        elif diff_percent < -0.15: # Predicted price is >15% higher than actual
            return "Overpriced ❌"
        else:
            return "Fair Price 👍"
    
    results_df['evaluation'] = results_df.apply(get_evaluation, axis=1)

    # --- Merge with Image URLs ---
    if image_map_df is not None:
        # Ensure the 'id' column in image_map_df is the same type as in results_df
        image_map_df['id'] = image_map_df['id'].astype(results_df['id'].dtype)
        results_df = pd.merge(results_df, image_map_df, on='id', how='left')
        # Fill missing image URLs with a default placeholder
        results_df['image_url'].fillna('https://example.com/images/default.jpg', inplace=True)
    else:
        results_df['image_url'] = None

    # Convert to the desired list of dictionaries (JSON format)
    output_json = results_df[[
        'id', 'brand', 'model', 'predicted_price', 'actual_price', 'evaluation', 'image_url'
    ]].to_dict(orient='records')

    # For demonstration, print the first 5 results in the new format
    print("\n--- Evaluation Results (JSON format sample) ---")
    import json
    print(json.dumps(output_json[:5], indent=2))

    return output_json

def predict_new_car(model, train_columns, car_details):
    """Predicts the price of a new car, ensuring columns match the training data."""
    new_car_df = pd.DataFrame([car_details])
    
    # The 'id' column is not a feature, so we should use columns from X_train
    train_features = [col for col in train_columns if col != 'id']
    # Align columns with the training data, filling missing ones with NaN
    new_car_df = new_car_df.reindex(columns=train_features, fill_value=np.nan)
    
    predicted_price = model.predict(new_car_df)
    
    print(f"\n--- New Car Prediction ---")
    print(f"Car Details: {car_details}")
    print(f"Predicted Price: {predicted_price[0]:,.2f}")
    
    # Create the JSON output
    output = {
        "id": car_details.get("id", None), # Use id if provided, else None
        "brand": car_details.get("brand"),
        "model": car_details.get("model"),
        "predicted_price": predicted_price[0],
        "actual_price": None, # Not applicable for a new prediction
        "evaluation": None # Not applicable for a new prediction
    }
    
    return output

if __name__ == "__main__":
    print("Script started...", flush=True)

    # Define paths for model and columns
    MODEL_PATH = 'car_price_model.pkl'
    COLUMNS_PATH = 'model_columns.json'

    # 1. Load and prepare data
    print("Step 1: Loading and cleaning data...", flush=True)
    df, target_col, year_col = load_and_clean_data("car_ads_details_kaggle.csv")
    print("Data loaded. Starting feature engineering...", flush=True)
    df = feature_engineering(df.copy(), year_col)
    print("Step 1 complete.", flush=True)

    # 2. Define features and target
    print("Step 2: Defining features and target...", flush=True)
    features = [c for c in df.columns if c not in [target_col]] # id is kept in X for now
    X = df[features]
    y = df[target_col]
    print("Step 2 complete.", flush=True)

    # 3. Identify feature types
    print("Step 3: Identifying feature types...", flush=True)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = [c for c in X.select_dtypes(include=['number']).columns.tolist() if c != 'id']
    print(f"Found {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features.", flush=True)
    print("Step 3 complete.", flush=True)

    # 4. Split data before model training/loading
    print("Step 4: Splitting data...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Step 4 complete.", flush=True)

    # 5. Train or Load Model
    if os.path.exists(MODEL_PATH):
        print(f"--- Loading existing model from {MODEL_PATH} ---", flush=True)
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.", flush=True)
    else:
        print(f"--- Model not found. Starting new training... ---", flush=True)
        # Build pipeline
        pipeline = build_pipeline(numeric_cols, categorical_cols)
        
        # Train model
        model = train_model(X_train, y_train, pipeline)
        
        # Save the trained model
        print(f"Saving model to {MODEL_PATH}...", flush=True)
        joblib.dump(model, MODEL_PATH)
        
        # Save the training columns for the backend
        print(f"Saving feature columns to {COLUMNS_PATH}...", flush=True)
        with open(COLUMNS_PATH, 'w') as f:
            json.dump(X_train.columns.tolist(), f, indent=4)

    # Load image mapping data
    try:
        image_map = pd.read_csv("cars_images.csv")
        print("Image mapping file loaded successfully.", flush=True)
    except FileNotFoundError:
        print("Warning: cars_images.csv not found. Proceeding without image URLs.", flush=True)
        image_map = None

    # 6. Evaluate model predictions
    print("\nStep 6: Evaluating model predictions...", flush=True)
    results_json = evaluate_predictions(model, X_test, y_test, image_map_df=image_map)
    print("Step 6 complete.", flush=True)

    # 7. Predict price for a new car example
    print("\nStep 7: Predicting price for a new car example...", flush=True)
    new_car = {
        'id': 'new_car_001', # Example ID for the new car
        'brand': 'Toyota',
        'model': 'Corolla',
        'mileage': 80000,
        'fuel': 'Petrol',
        'transmission': 'Automatic',
    }
    # Add car_age to the example only if it was used as a feature
    if 'car_age' in X_train.columns:
        new_car['car_age'] = 5 # Example: 2026 - 5 = 2021 model
    
    # Use X_train.columns to ensure consistency
    prediction_output = predict_new_car(model, X_train.columns, new_car)
    predicted_price = prediction_output['predicted_price']

    # Example of comparing with an actual price
    actual_price = 450000
    difference = actual_price - predicted_price
    
    prediction_output['actual_price'] = actual_price
    print(f"Actual Price: {actual_price:,.2f}", flush=True)
    if actual_price > 0:
        diff_percent = difference / actual_price
        if diff_percent > 0.15:
            prediction_output['evaluation'] = "Good Deal 🔥"
        elif diff_percent < -0.15:
            prediction_output['evaluation'] = "Overpriced ❌"
        else:
            prediction_output['evaluation'] = "Fair Price 👍"
    print(f"Conclusion: {prediction_output['evaluation']}", flush=True)
    print("Final JSON output for new car:", prediction_output)
    print("Step 7 complete.", flush=True)
    print("Script finished.", flush=True)
