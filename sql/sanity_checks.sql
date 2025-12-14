-- Verify rows exist and inspect columns
SELECT * FROM activity LIMIT 5;
SELECT * FROM energy LIMIT 5;
SELECT * FROM heart_rate LIMIT 5;

-- Example: check date ranges
SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM activity;

-- Example: count NULLs in key columns
SELECT
  SUM(CASE WHEN steps IS NULL THEN 1 ELSE 0 END) AS null_steps
FROM activity;
