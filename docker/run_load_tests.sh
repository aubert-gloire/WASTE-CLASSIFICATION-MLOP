#!/bin/bash

# Docker Load Testing Automation Script
# Runs load tests with 1, 2, 3, and 5 containers automatically

set -e

echo "=========================================="
echo "Waste Classification Load Testing Suite"
echo "=========================================="
echo ""

# Create results directory
mkdir -p results

# Function to wait for containers to be healthy
wait_for_health() {
    echo "Waiting for containers to be healthy..."
    sleep 30
    
    max_attempts=10
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✓ Containers are healthy"
            return 0
        fi
        echo "Attempt $((attempt+1))/$max_attempts - waiting..."
        sleep 10
        attempt=$((attempt+1))
    done
    
    echo "✗ Containers failed to become healthy"
    return 1
}

# Scenario 1: Single Container
echo ""
echo "=========================================="
echo "SCENARIO 1: Single Container (Baseline)"
echo "=========================================="

cd docker
docker-compose up -d api-1
cd ..

wait_for_health

echo "Running load test (50 users, 5 min)..."
locust -f locustfile.py \
    --host=http://localhost:8000 \
    --users 50 \
    --spawn-rate 5 \
    --run-time 5m \
    --headless \
    --csv=results/1-container \
    --html=results/1-container.html

echo "Collecting container stats..."
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" > results/1-container-stats.txt

cd docker
docker-compose down
cd ..

echo "✓ Scenario 1 complete"
sleep 10

# Scenario 2: Two Containers
echo ""
echo "=========================================="
echo "SCENARIO 2: Two Containers with Load Balancer"
echo "=========================================="

cd docker
docker-compose -f docker-compose.loadtest.yml up -d nginx api-1 api-2
cd ..

wait_for_health

echo "Running load test (100 users, 5 min)..."
locust -f locustfile.py \
    --host=http://localhost:8000 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 5m \
    --headless \
    --csv=results/2-containers \
    --html=results/2-containers.html

echo "Collecting container stats..."
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" > results/2-containers-stats.txt

cd docker
docker-compose -f docker-compose.loadtest.yml down
cd ..

echo "✓ Scenario 2 complete"
sleep 10

# Scenario 3: Three Containers
echo ""
echo "=========================================="
echo "SCENARIO 3: Three Containers with Load Balancer"
echo "=========================================="

cd docker
docker-compose -f docker-compose.loadtest.yml up -d
cd ..

wait_for_health

echo "Running load test (150 users, 5 min)..."
locust -f locustfile.py \
    --host=http://localhost:8000 \
    --users 150 \
    --spawn-rate 15 \
    --run-time 5m \
    --headless \
    --csv=results/3-containers \
    --html=results/3-containers.html

echo "Collecting container stats..."
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" > results/3-containers-stats.txt

cd docker
docker-compose -f docker-compose.loadtest.yml down
cd ..

echo "✓ Scenario 3 complete"
sleep 10

# Generate Summary Report
echo ""
echo "=========================================="
echo "Generating Summary Report"
echo "=========================================="

cat > results/SUMMARY.md << 'EOF'
# Load Testing Results Summary

## Test Configuration

- **Tool:** Locust
- **Test Duration:** 5 minutes per scenario
- **Date:** $(date)

## Scenarios Tested

### Scenario 1: Single Container (Baseline)
- **Users:** 50 concurrent
- **Spawn Rate:** 5 users/second
- **Containers:** 1

### Scenario 2: Two Containers
- **Users:** 100 concurrent
- **Spawn Rate:** 10 users/second
- **Containers:** 2 (with Nginx load balancer)

### Scenario 3: Three Containers
- **Users:** 150 concurrent
- **Spawn Rate:** 15 users/second
- **Containers:** 3 (with Nginx load balancer)

## Results Comparison

| Metric | 1 Container | 2 Containers | 3 Containers | Improvement |
|--------|-------------|--------------|--------------|-------------|
| Total Requests | - | - | - | - |
| Requests/sec | - | - | - | - |
| Avg Response (ms) | - | - | - | - |
| Median Response (ms) | - | - | - | - |
| 95th Percentile (ms) | - | - | - | - |
| Failure Rate (%) | - | - | - | - |

*Fill in values from CSV files: 1-container_stats.csv, 2-containers_stats.csv, 3-containers_stats.csv*

## Key Findings

1. **Throughput Scaling:**
   - 2 containers vs 1: X% improvement
   - 3 containers vs 1: Y% improvement

2. **Latency Reduction:**
   - 2 containers vs 1: X% faster
   - 3 containers vs 1: Y% faster

3. **Reliability:**
   - Failure rate decreased by X% with load balancing

## Conclusion

[Add your analysis here]

## Files Generated

- `1-container_stats.csv` - Detailed metrics for single container
- `2-containers_stats.csv` - Detailed metrics for two containers
- `3-containers_stats.csv` - Detailed metrics for three containers
- `*.html` - Interactive HTML reports for each scenario
- `*-stats.txt` - Docker container resource usage

EOF

echo "✓ Summary report template created: results/SUMMARY.md"

# List all generated files
echo ""
echo "=========================================="
echo "Generated Files"
echo "=========================================="
ls -lh results/

echo ""
echo "=========================================="
echo "All Load Tests Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review results in results/ directory"
echo "2. Fill in SUMMARY.md with actual metrics from CSV files"
echo "3. Include charts/screenshots in your assignment"
echo "4. Add findings to README.md"
echo ""
