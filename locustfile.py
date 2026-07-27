from pathlib import Path
from locust import HttpUser, task, between

# Path to our test image
IMAGE_PATH = Path("wildfire-dataset/test/wildfire/-79.403,48.85132.jpg")

class WildfireApiUser(HttpUser):
    # Simulate a user waiting 0.5 to 2.0 seconds between requests
    wait_time = between(0.5, 2.0)

    @task
    def predict_wildfire(self):
        with open(IMAGE_PATH, "rb") as img:
            files = {"file": (IMAGE_PATH.name, img, "image/jpeg")}
            self.client.post("/predict?threshold=0.4", files=files)