import tritonclient.http as httpclient

print("Triton Inference Server Intermediate Tutorial")
print("==============================================")

# 1. Triton Client Initialization and Model Inference
print("\n1. Triton Client Initialization and Model Inference:")
client = httpclient.InferenceServerClient(url="localhost:8000")

# 2. Sending Inference Requests
print("\n2. Sending Inference Requests:")
# Placeholder for sending inference requests code
