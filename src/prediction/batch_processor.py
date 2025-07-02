import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
import pickle

class BatchProcessor:
    """Batch processing engine for credit default predictions"""

    def __init__(self, model_path: str = None, max_workers: int = 4):
        self.model = None
        self.model_path = model_path
        self.max_workers = max_workers
        self.progress = {"current": 0, "total": 0, "status": "idle"}
        self.lock = threading.Lock()
        self.results = None

    def load_model(self, model_path: str = None):
        """Load the trained model"""
        path = model_path or self.model_path
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            return True
        return False

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate uploaded batch file"""
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "file_info": {}
        }

        try:
            if not os.path.exists(file_path):
                validation_result["errors"].append("File not found")
                return validation_result

            # Check file extension
            if not file_path.lower().endswith(('.csv', '.xlsx', '.xls')):
                validation_result["errors"].append("Unsupported file format. Use CSV or Excel files.")
                return validation_result

            # Load and validate data
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            validation_result["file_info"] = {
                "rows": len(df),
                "columns": len(df.columns),
                "size_mb": round(os.path.getsize(file_path) / (1024*1024), 2)
            }

            # Required columns check
            required_cols = ['age', 'income', 'loan_amount', 'credit_score', 'employment_length']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                validation_result["errors"].append(f"Missing required columns: {missing_cols}")

            # Data quality checks
            if df.isnull().sum().sum() > 0:
                validation_result["warnings"].append("Dataset contains missing values")

            if len(df) == 0:
                validation_result["errors"].append("File is empty")
            elif len(df) > 10000:
                validation_result["warnings"].append("Large dataset may take longer to process")

            validation_result["valid"] = len(validation_result["errors"]) == 0

        except Exception as e:
            validation_result["errors"].append(f"File validation error: {str(e)}")

        return validation_result

    def preprocess_batch_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess batch data for prediction"""
        processed_df = df.copy()

        # Handle missing values
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())

        # Handle categorical columns
        categorical_cols = processed_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            processed_df[col] = processed_df[col].fillna('Unknown')

        # Feature engineering
        if 'income' in processed_df.columns and 'loan_amount' in processed_df.columns:
            processed_df['debt_to_income'] = processed_df['loan_amount'] / (processed_df['income'] + 1)

        if 'age' in processed_df.columns:
            processed_df['age_group'] = pd.cut(processed_df['age'], 
                                             bins=[0, 25, 35, 50, 100], 
                                             labels=['Young', 'Adult', 'Middle', 'Senior'])

        return processed_df

    def predict_batch_chunk(self, chunk: pd.DataFrame, chunk_id: int) -> Dict[str, Any]:
        """Process a single chunk of data"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")

            # Preprocess chunk
            processed_chunk = self.preprocess_batch_data(chunk)

            # Select features for prediction
            feature_cols = ['age', 'income', 'loan_amount', 'credit_score', 'employment_length', 'debt_to_income']
            available_cols = [col for col in feature_cols if col in processed_chunk.columns]

            if not available_cols:
                raise ValueError("No valid features found for prediction")

            X = processed_chunk[available_cols]

            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None

            # Update progress
            with self.lock:
                self.progress["current"] += len(chunk)

            return {
                "chunk_id": chunk_id,
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist() if probabilities is not None else None,
                "processed_rows": len(chunk)
            }

        except Exception as e:
            return {
                "chunk_id": chunk_id,
                "error": str(e),
                "processed_rows": 0
            }

    def process_batch_file(self, file_path: str, chunk_size: int = 1000) -> Dict[str, Any]:
        """Process entire batch file with parallel processing"""
        try:
            # Load data
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # Initialize progress
            with self.lock:
                self.progress = {"current": 0, "total": len(df), "status": "processing"}

            # Split into chunks
            chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

            # Process chunks in parallel
            results = []
            errors = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_chunk = {
                    executor.submit(self.predict_batch_chunk, chunk, i): i 
                    for i, chunk in enumerate(chunks)
                }

                for future in as_completed(future_to_chunk):
                    result = future.result()
                    if "error" in result:
                        errors.append(result)
                    else:
                        results.append(result)

            # Combine results
            all_predictions = []
            all_probabilities = []

            for result in sorted(results, key=lambda x: x["chunk_id"]):
                all_predictions.extend(result["predictions"])
                if result["probabilities"]:
                    all_probabilities.extend(result["probabilities"])

            # Create results dataframe
            results_df = df.copy()
            results_df['prediction'] = all_predictions[:len(df)]
            results_df['default_risk'] = ['High' if p == 1 else 'Low' for p in all_predictions[:len(df)]]

            if all_probabilities:
                results_df['probability'] = all_probabilities[:len(df)]
                results_df['risk_score'] = pd.cut(all_probabilities[:len(df)], 
                                                bins=[0, 0.3, 0.7, 1.0], 
                                                labels=['Low', 'Medium', 'High'])

            self.results = results_df

            with self.lock:
                self.progress["status"] = "completed"

            return {
                "success": True,
                "processed_rows": len(df),
                "errors": errors,
                "summary": self.generate_summary()
            }

        except Exception as e:
            with self.lock:
                self.progress["status"] = "error"
            return {"success": False, "error": str(e)}

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary analytics"""
        if self.results is None:
            return {}

        df = self.results
        summary = {
            "total_records": len(df),
            "high_risk_count": sum(df['prediction'] == 1),
            "low_risk_count": sum(df['prediction'] == 0),
            "high_risk_percentage": round(sum(df['prediction'] == 1) / len(df) * 100, 2)
        }

        if 'probability' in df.columns:
            summary.update({
                "avg_risk_probability": round(df['probability'].mean(), 3),
                "max_risk_probability": round(df['probability'].max(), 3),
                "min_risk_probability": round(df['probability'].min(), 3)
            })

        # Risk distribution
        if 'risk_score' in df.columns:
            risk_dist = df['risk_score'].value_counts().to_dict()
            summary["risk_distribution"] = risk_dist

        return summary

    def export_results(self, output_dir: str, formats: List[str] = ['csv']) -> Dict[str, str]:
        """Export results in multiple formats"""
        if self.results is None:
            return {"error": "No results to export"}

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files = {}

        try:
            for format_type in formats:
                if format_type.lower() == 'csv':
                    file_path = os.path.join(output_dir, f"batch_predictions_{timestamp}.csv")
                    self.results.to_csv(file_path, index=False)
                    exported_files['csv'] = file_path

                elif format_type.lower() == 'excel':
                    file_path = os.path.join(output_dir, f"batch_predictions_{timestamp}.xlsx")
                    self.results.to_excel(file_path, index=False)
                    exported_files['excel'] = file_path

                elif format_type.lower() == 'json':
                    file_path = os.path.join(output_dir, f"batch_predictions_{timestamp}.json")
                    self.results.to_json(file_path, orient='records', indent=2)
                    exported_files['json'] = file_path

            # Export summary
            summary_path = os.path.join(output_dir, f"batch_summary_{timestamp}.json")
            with open(summary_path, 'w') as f:
                json.dump(self.generate_summary(), f, indent=2)
            exported_files['summary'] = summary_path

            return exported_files

        except Exception as e:
            return {"error": f"Export failed: {str(e)}"}

    def get_progress(self) -> Dict[str, Any]:
        """Get current processing progress"""
        with self.lock:
            progress_copy = self.progress.copy()

        if progress_copy["total"] > 0:
            progress_copy["percentage"] = round(progress_copy["current"] / progress_copy["total"] * 100, 1)
        else:
            progress_copy["percentage"] = 0

        return progress_copy
