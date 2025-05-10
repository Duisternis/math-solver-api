import requests

try:
    response = requests.post(
        "http://127.0.0.1:8000/solve",
        json={"expression": "x^2 - 4 = 0"},
        stream=True
    )

    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))

except requests.exceptions.ChunkedEncodingError:
    print("✅ Stream ended (likely SSE stream closed cleanly)")
