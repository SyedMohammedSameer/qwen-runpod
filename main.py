import runpod
from handler import EndpointHandler

# Load the model once when the worker starts
extractor = EndpointHandler()

def handler(job):
    """Handle incoming RunPod serverless requests."""
    inputs = job["input"]
    result = extractor(inputs)
    return result

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})