# Weather Harvester

I fetch current weather from selected cities and save them to `.tab` for future analysis. Data include temperature (celcius), humidity (%), rain (mm), time (UNIX UTC).

## Configuration File

Please refer to `example.toml` for an example configuration file.

The script expects the configuration file to be at `config.toml` in the current directory. You can override it with the `CONFIG_FILE` environment variable.

For example

```bash
CONFIG_FILE="/path/to/config.toml" "API Weather.py"
```

## API key generation

Make an account at [OpenWeatherMap](https://home.openweathermap.org/api_keys).

Go to the API keys tab and generate a key which will be in the format
```toml
api_key = "bb0664ed43c153aa072c760594d775a7"
```

## Find city ID

Under `Place_id` in configuration file `example.toml`, enter the city ID as per "city.list.json.gz" at [OpenWeatherMaps](http://bulk.openweathermap.org/sample/). Unzip the `.gz` file to get a `.json` file in a text editor and search for your cities of interest then copy the dictionary value from the key `id`.

E.g.
```json
{
    "id": 1894616,
    "name": "Okinawa",
    "country": "JP",
    "coord": {
      "lon": 127.801392,
      "lat": 26.335831
    }
}
```

The id value is to be placed here:

```toml
Place_id = [
    "1894616", "6433095"
]
```

## Save Directory

Insert folder directory where you wish to have your `.tab` file(s) saved. The program expects a `.tab` and may error if you ask for a `.csv`.

```toml
saveme = "insert_directory_here"

tab_extension = ".tab"
```
