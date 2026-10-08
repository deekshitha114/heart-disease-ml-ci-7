
import json
import sys

with open("model_metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]
minimum_accuracy = 0.75

print("Model accuracy:", accuracy)
print("Required accuracy:", minimum_accuracy)

if accuracy < minimum_accuracy:
    print("FAIL: Model quality gate failed.")
    sys.exit(1)

print("PASS: Model quality gate passed.")
