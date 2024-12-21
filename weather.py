import requests
import aiohttp
import asyncio

API_KEY = "afb6147ee48eace31b567b026d07535e"


def get_current_temperature_sync(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data['main']['temp']
    elif response.status_code == 401:
        raise Exception("Error 401: Unauthorized. Please check your API key.")
    else:
        raise Exception(f"Error fetching data: {response.status_code}")

async def get_current_temperature_async(city, api_key=API_KEY):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data['main']['temp']
            else:
                raise Exception(f"Error fetching data: {response.status}")