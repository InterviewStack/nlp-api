import requests
import json
import concurrent.futures
import time
from faker import Faker

# URL and JSON payload
url = 'http://localhost:8000/answer/'

def json_body():
    fake = Faker()
    model_answer = " ".join(fake.words(nb=200))
    user_answer = " ".join(fake.words(nb=200))
    json = {
        "model_answer": model_answer,
        "user_answer": user_answer
    }
    return json


# Function to make a GET request
def make_request():
    print("made request")
    response = requests.get(url, json=json_body())
    print(response.content)
    return response.elapsed.total_seconds()

# Load test function
def load_test(num_requests):
    response_times = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2000) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            response_times.append(future.result())

    min_response_time = min(response_times)
    max_response_time = max(response_times)
    mean_response_time = sum(response_times) / len(response_times)

    print(f"Min response time: {min_response_time} seconds")
    print(f"Max response time: {max_response_time} seconds")
    print(f"Mean response time: {mean_response_time} seconds")

# Entry point
if __name__ == "__main__":
    number_of_requests = 2000
    load_test(number_of_requests)
