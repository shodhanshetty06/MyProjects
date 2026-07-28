-- Dedicated database for the pytest suite, isolated from the app's
-- dev/prod database so test fixtures never leak into real search results.
SELECT 'CREATE DATABASE jobplatform_test OWNER jobplatform'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'jobplatform_test')\gexec

\c jobplatform_test
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
