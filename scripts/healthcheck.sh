#!/bin/bash
# EasyMLOps Health Check Script

set -e

echo "Running EasyMLOps Health Check..."

# Check application health endpoint
echo "Checking application health..."
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")
if [ "$response" = "200" ]; then
    echo "✓ Application is healthy"
else
    echo "✗ Application health check failed (HTTP $response)"
    exit 1
fi

# Check database connectivity
echo "Checking database connectivity..."
python -c "
import sys
from sqlalchemy import create_engine, text
from app.core.config import get_settings

try:
    settings = get_settings()
    engine = create_engine(str(settings.database_url))
    with engine.connect() as connection:
        result = connection.execute(text('SELECT 1'))
        print('✓ Database is accessible')
except Exception as e:
    print(f'✗ Database check failed: {e}')
    sys.exit(1)
"

# Check critical API endpoints
echo "Checking critical API endpoints..."
endpoints=(
    "/api/v1/models"
    "/api/v1/deployments" 
    "/api/v1/monitoring/health"
)

for endpoint in "${endpoints[@]}"; do
    response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000$endpoint" || echo "000")
    if [ "$response" = "200" ] || [ "$response" = "401" ]; then
        echo "✓ Endpoint $endpoint is responding"
    else
        echo "✗ Endpoint $endpoint failed (HTTP $response)"
        exit 1
    fi
done

# Check system resources
echo "Checking system resources..."
python -c "
import psutil
import sys

# Check memory usage
memory = psutil.virtual_memory()
if memory.percent < 90:
    print(f'✓ Memory usage: {memory.percent:.1f}%')
else:
    print(f'⚠ High memory usage: {memory.percent:.1f}%')

# Check disk usage
disk = psutil.disk_usage('/')
if disk.percent < 90:
    print(f'✓ Disk usage: {disk.percent:.1f}%')
else:
    print(f'⚠ High disk usage: {disk.percent:.1f}%')

# Check CPU usage
cpu_percent = psutil.cpu_percent(interval=1)
if cpu_percent < 80:
    print(f'✓ CPU usage: {cpu_percent:.1f}%')
else:
    print(f'⚠ High CPU usage: {cpu_percent:.1f}%')
"

echo "Health check completed successfully!"
exit 0 