#!/bin/bash
# EasyMLOps Application Startup Script

set -e

echo "Starting EasyMLOps Application..."

# Wait for database to be ready
echo "Waiting for database to be ready..."
python -c "
import time
import sys
from sqlalchemy import create_engine
from app.config import get_settings

settings = get_settings()
max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        engine = create_engine(str(settings.DATABASE_URL))
        connection = engine.connect()
        connection.close()
        print('Database is ready!')
        break
    except Exception as e:
        retry_count += 1
        print(f'Database not ready, retrying... ({retry_count}/{max_retries})')
        time.sleep(2)
        if retry_count >= max_retries:
            print('Database connection failed after maximum retries')
            sys.exit(1)
"

# Run database migrations
echo "Running database migrations..."
python -m alembic upgrade head || echo "Migrations skipped or already applied"

# Start the application
echo "Starting application server..."
exec python -m app.main --host 0.0.0.0 --port 8000 