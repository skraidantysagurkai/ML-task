"""
    This is a unit test for Endpoint.py
"""
import requests
import json
import unittest


class TestEndpoint(unittest.TestCase):
    url = None

    @classmethod
    def setUpClass(cls):
        """
            This functions setups class variables we will need.
        """
        # Initialize the shared_variable before any test method runs
        cls.url = 'http://127.0.0.1:5000'

    def test_upper(self):
        data = {'row1': {'LP': 22.203,
                         'Continent': 'Asia',
                         'NGSD_NGDP': '61.151',
                         'LE': 2.41,
                         'BCA': 1.481,
                         'GGR_NGDP': 6.845,
                         'LUR': 42.4,
                         'GGSB_NPGDP': 6}
                }
        json_string = json.dumps(data)
        response = requests.post(self.url, json=json_string)

        self.assertEqual(response.status_code, 200)

    def test_isupper(self):
        data = {'row1': {'LP': 22.203,
                         'Continent': 'Asia',
                         'NGSD_NGDP': '61.151',
                         'LE': 2.41,
                         'BCA': 1.481,
                         'GGR_NGDP': 6.845,
                         'LUR': 42.4,
                         'GGSB_NPGDP': 6}
                }
        json_string = json.dumps(data)
        response = requests.post(self.url, json=json_string)

        self.assertEqual(response.status_code, 200)

    def test_split(self):
        data = {'row1': {'LP': 22.203,
                         'Continent': 'Asia',
                         'NGSD_NGDP': '61.151',
                         'LE': 2.41,
                         'BCA': 1.481,
                         'GGR_NGDP': 6.845,
                         'LUR': 42.4,
                         'GGSB_NPGDP': 6}
                }
        json_string = json.dumps(data)
        response = requests.post(self.url, json=json_string)

        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
