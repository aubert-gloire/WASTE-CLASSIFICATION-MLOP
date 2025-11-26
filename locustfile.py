"""
Locust Load Testing Configuration for Waste Classification API
Simulates flood of requests to test scalability and performance
"""
from locust import HttpUser, task, between
import random
import io
from PIL import Image
import numpy as np


class WasteClassificationUser(HttpUser):
    """Simulated user for load testing"""
    
    # Wait between 1-3 seconds between tasks
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user session"""
        # Check API health
        response = self.client.get("/health")
        if response.status_code != 200:
            print("Warning: API health check failed")
    
    @task(10)
    def predict_image(self):
        """
        Task: Predict waste class for an image
        Weight: 10 (most common operation)
        """
        # Generate a random test image
        img = self._generate_random_image()
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Send prediction request
        files = {'file': ('test_image.jpg', img_byte_arr, 'image/jpeg')}
        
        with self.client.post(
            "/predict",
            files=files,
            catch_response=True,
            name="/predict [single image]"
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if 'predicted_class' in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(5)
    def get_health(self):
        """
        Task: Check API health
        Weight: 5
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(3)
    def get_metrics(self):
        """
        Task: Get system metrics
        Weight: 3
        """
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def get_model_info(self):
        """
        Task: Get model information
        Weight: 2
        """
        with self.client.get("/model/info", catch_response=True) as response:
            if response.status_code == 200 or response.status_code == 503:
                # 503 is acceptable if model not loaded
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def predict_batch(self):
        """
        Task: Predict batch of images
        Weight: 1 (less common, more resource-intensive)
        """
        num_images = random.randint(2, 5)
        files = []
        
        for i in range(num_images):
            img = self._generate_random_image()
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            files.append(('files', (f'test_image_{i}.jpg', img_byte_arr, 'image/jpeg')))
        
        with self.client.post(
            "/predict/batch",
            files=files,
            catch_response=True,
            name=f"/predict/batch [{num_images} images]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    def _generate_random_image(self, size=(224, 224)):
        """Generate a random test image"""
        # Create random RGB image
        arr = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
        img = Image.fromarray(arr, 'RGB')
        return img


class HeavyLoadUser(HttpUser):
    """User that generates heavy load with rapid requests"""
    
    wait_time = between(0.1, 0.5)  # Very short wait time
    
    @task
    def rapid_predictions(self):
        """Make rapid prediction requests"""
        img = Image.new('RGB', (224, 224), color=(random.randint(0, 255), 
                                                    random.randint(0, 255), 
                                                    random.randint(0, 255)))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
        self.client.post("/predict", files=files)


class StressTestUser(HttpUser):
    """User for stress testing with larger payloads"""
    
    wait_time = between(2, 5)
    
    @task
    def large_batch_prediction(self):
        """Send large batch of images"""
        num_images = random.randint(10, 20)
        files = []
        
        for i in range(num_images):
            # Larger images
            img = Image.new('RGB', (512, 512), 
                          color=(random.randint(0, 255), 
                                random.randint(0, 255), 
                                random.randint(0, 255)))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            files.append(('files', (f'large_image_{i}.jpg', img_byte_arr, 'image/jpeg')))
        
        self.client.post("/predict/batch", files=files, timeout=60)


# Command line usage examples:
"""
Basic load test (10 users, 2 users/second spawn rate):
    locust -f locustfile.py --host=http://localhost:8000 --users 10 --spawn-rate 2 --run-time 5m

Heavy load test (100 users, 10 users/second):
    locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 10m

Stress test (500 users, 50 users/second):
    locust -f locustfile.py --host=http://localhost:8000 --users 500 --spawn-rate 50 --run-time 15m

Specific user class:
    locust -f locustfile.py --host=http://localhost:8000 WasteClassificationUser --users 50 --spawn-rate 5

Headless mode (no web UI):
    locust -f locustfile.py --host=http://localhost:8000 --headless --users 100 --spawn-rate 10 --run-time 5m

With CSV output:
    locust -f locustfile.py --host=http://localhost:8000 --headless --users 100 --spawn-rate 10 --run-time 5m --csv=results

Test against deployed API:
    locust -f locustfile.py --host=http://your-api-url.com --users 100 --spawn-rate 10 --run-time 10m
"""
