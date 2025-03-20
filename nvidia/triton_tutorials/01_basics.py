import tritonclient.http as httpclient

print("Triton Inference Server Basics Tutorial")
print("======================================")

# 1. Triton Client Initialization
print("\n1. Triton Client Initialization:")
client = httpclient.InferenceServerClient(url="localhost:8000")

# 2. Model Availability
print("\n2. Checking Model Availability:")
model_name = "simple_model"
if client.is_model_ready(model_name):
    print(f"Model {model_name} is ready.")
else:
    print(f"Model {model_name} is not ready.")

print("\nNote: This tutorial demonstrates basic Triton client usage. Ensure Triton server is running and accessible.")
