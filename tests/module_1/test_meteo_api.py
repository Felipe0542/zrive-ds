""" This is a dummy example to show how to import code from src/ for testing"""

from src.module_1.module_1_meteo_api import main, modify_weather_data, call_api
import pytest
import pandas as pd
from unittest.mock import Mock, patch #tal vez quitar
import unittest
import requests

class MockResponse:
    def __init__(self, status_code, json_data = None):
        self.json_data = json_data or {}
        self.status_code = status_code
    
    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.exceptions.HTTPError(f"Error HPPT: {self.status_code}")

class TestCallApi(unittest.TestCase):
    headers = {}

    def test_modify_weather_data(self): #no se usa para nada el self pero no me funcionaba si no lo metía en la clase y para que funcione en la clase he tenido que ponerlo
        data = pd.DataFrame({
            "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "temperature_2m_mean": [10, 15, 20],
            "precipitation_sum": [5, 3, 2],
            "wind_speed_10m_max": [30, 40, 35],
            "city": ["Madrid", "Madrid", "Madrid"],
        })

        variables = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
        result = modify_weather_data(data, variables)

        assert set(result.columns) == {"city", "month", "temperature_2m_mean_mean", "precipitation_sum_mean", "wind_speed_10m_max_mean"}
        assert result.loc[result["city"] == "Madrid", "temperature_2m_mean_mean"].values[0] == 15  # media bien

    @patch('src.module_1.module_1_meteo_api.requests.get')

    def test_call_api_200(self, mock_get):
        '''
        çmocked_response = Mock(return_value = MockResponse("json_test", 200))
        monkeypatch.setattr(requests, "get", mocked_response)
        response = call_api("URL", headers, 10)
        assert response.status_code == 200
        assert response.json()  = "json_test"
        '''
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"example": "example"}

        response = call_api("example/api", headers={})
        self.assertEqual(response, {"example": "example"})
        self.assertEqual(mock_get.return_value.status_code, 200)
        mock_get.assert_called_once()

    @patch('src.module_1.module_1_meteo_api.requests.get')

    def test_call_api_404(self, mock_get):
        
        mock_get.side_effect = requests.exceptions.HTTPError("404 Client Error: Not Found")
        response = call_api("example/api", headers={})
        self.assertIsNone(response)
        mock_get.assert_called_with("example/api", headers={})

    @patch('src.module_1.module_1_meteo_api.requests.get')

    def test_call_api_429(self, mock_get):
        mock_get.side_effect = requests.exceptions.HTTPError("429 Too Many Requests")

        response = call_api("example/api", headers={})
        self.assertIsNone(response)
        mock_get.assert_called_with("example/api", headers={})

def main():
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))

    
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    
    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)

if __name__ == '__main__':
    main()