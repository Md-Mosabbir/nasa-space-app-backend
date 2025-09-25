import requests
import json

def fetch_power_api(temporal, start, end, latitude, longitude, parameters):
	"""
	Fetch data from NASA PowerAPI and return JSON response.
	Args:
		temporal (str): 'daily', 'hourly', etc.
		start (int): Start year (e.g., 2020)
		end (int): End year (e.g., 2025)
		latitude (float): Latitude value
		longitude (float): Longitude value
		parameters (list): List of parameter strings (e.g., ['T2M', 'PRECTOT'])
	Returns:
		dict: JSON response from API
	"""
	base_url = "https://power.larc.nasa.gov/api/temporal/{}/point?parameters={}&community=RE&longitude={}&latitude={}&start={}&end={}&format=JSON"
	query_url = base_url.format(temporal, ','.join(parameters), longitude, latitude, start, end)
	response = requests.get(url=query_url, verify=True, timeout=30)
	return response.json()

if __name__ == "__main__":
	# Example usage and print a single data value
	temporal = 'daily'
	start = 2025
	end = 2025
	latitude = 23.777176
	longitude = 90.399452
	parameters = ['T2M', 'PRECTOT', 'WS10M', 'RH2M']
	result = fetch_power_api(temporal, start, end, latitude, longitude, parameters)
	# Print a single value to check
	try:
		print(result['properties']['parameter']['T2M'])
	except Exception as e:
		print("Error printing T2M data:", e)
