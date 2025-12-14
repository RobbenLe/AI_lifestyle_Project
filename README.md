# AI Health Model (PostgreSQL + PyTorch)

This project trains a small neural network on daily health features aggregated from PostgreSQL.

## Getting started

1. Create your environment (or reuse your **AE Shared** venv) and install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set your PostgreSQL credentials:
   ```ini
   PGHOST=localhost
   PGPORT=5432
   PGDATABASE=analytics
   PGUSER=postgres
   PGPASSWORD=postgres
   ```

3. Run quick SQL sanity checks (optional): see `sql/sanity_checks.sql`.

4. Use the scripts:
   - `python -m scripts.export_features` to export a CSV of daily features.
   - `python -m scripts.predict_next_day --date YYYY-MM-DD` to get a next-day prediction (after training).

## Structure

See the folder layout inside the repository. Each Python file is heavily commented and beginner-friendly.
