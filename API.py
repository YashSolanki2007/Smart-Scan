# Imports
import requests

# Specifying the URL
URL = "https://api.covid19api.com/summary"

# Getting the response
response = requests.get(URL)

# Converting the response to json format
jsonated_response = response.json()

total_cases = jsonated_response['Global']['TotalConfirmed']
total_deaths = jsonated_response['Global']['TotalDeaths']

print(f"Total cases in the world {total_cases}")
print(f"Total deaths in the world {total_deaths}")


# For the country specific
country_inp = input("Enter country name: ")

countries_list = jsonated_response['Countries']

for i in range(len(countries_list)):
    country_name_in_response = countries_list[i]['Country']
    country_total_cases = countries_list[i]['TotalConfirmed']
    country_total_deaths = countries_list[i]['TotalConfirmed']
    country_new_cases = countries_list[i]['NewConfirmed']
    country_new_deaths = countries_list[i]['NewDeaths']

    if country_inp == country_name_in_response:
        print(f"Total Confirmed: {country_total_cases} and Total Deaths: {country_total_deaths} New cases: {country_new_cases} New deaths: {country_new_deaths}")

    else:
        pass
