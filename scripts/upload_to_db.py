import os
import argparse
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def upload_to_db(fresh=False):
    # Load environment variables
    load_dotenv()
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        print("Error: POSTGRES_URL not found in .env")
        return

    # Create engine
    engine = create_engine(postgres_url)
    
    # Mapping of parquet files to table names
    table_mapping = {
        "dim_item.parquet": "dim_item",
        "dim_model.parquet": "dim_model",
        "fact_session_summary.parquet": "fact_session_summary",
        "fact_purchases.parquet": "fact_purchases",
        "fact_test_examples.parquet": "fact_test_examples",
        "fact_recommendations.parquet": "fact_recommendations",
        "fact_recommendation_eval.parquet": "fact_recommendation_eval",
        "fact_metrics.parquet": "fact_metrics",
        "fact_marketing_kpis.parquet": "fact_marketing_kpis",
    }

    # 0. Fresh Recreate if requested
    if fresh:
        print("Fresh flag detected. Dropping existing tables...")
        with engine.connect() as conn:
            # Drop tables in reverse to respect potential (though not explicit in schema) dependencies
            tables_to_drop = list(table_mapping.values())
            for table_name in reversed(tables_to_drop):
                print(f"Dropping table {table_name}...")
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
            conn.commit()
        print("Tables dropped successfully.")

    # 1. Apply Schema
    schema_path = Path("sql/schema.sql")
    if schema_path.exists():
        print(f"Applying schema from {schema_path}...")
        with open(schema_path, "r") as f:
            schema_sql = f.read()
        
        with engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()
        print("Schema applied successfully.")
    else:
        print(f"Warning: {schema_path} not found.")

    # 2. Upload Data
    mart_root = Path("data/mart")
    if not mart_root.exists():
        print("Error: data/mart directory not found. Run 'make marts' first.")
        return

    for parquet_file, table_name in table_mapping.items():
        file_path = mart_root / parquet_file
        if file_path.exists():
            print(f"Uploading {parquet_file} to table {table_name}...")
            df = pd.read_parquet(file_path)
            
            try:
                # Get the actual columns from the table to avoid uploading extra columns
                with engine.connect() as conn:
                    result = conn.execute(text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"))
                    db_columns = [row[0] for row in result]
                
                # Filter dataframe to only include columns that exist in the database
                upload_df = df[[col for col in df.columns if col in db_columns]]
                
                upload_df.to_sql(table_name, engine, if_exists='append', index=False)
                print(f"Successfully uploaded {len(upload_df)} rows to {table_name}.")
            except Exception as e:
                print(f"Error uploading {table_name}: {e}")
        else:
            print(f"Skipping {parquet_file} (not found).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload DDM analytics marts to PostgreSQL.")
    parser.add_argument("--fresh", action="store_true", help="Drop existing tables before uploading.")
    args = parser.parse_args()
    
    upload_to_db(fresh=args.fresh)
