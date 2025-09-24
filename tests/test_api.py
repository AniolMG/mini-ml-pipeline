import requests
#
def test_prediction_endpoint():
    payload = {"Age": 29, "Sex": 1, "Pclass": 3}
    r = requests.post("http://127.0.0.1:8000/predict", json=payload)
    assert r.status_code == 200
    result = r.json()
    assert "prediction" in result