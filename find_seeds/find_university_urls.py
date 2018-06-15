from bs4 import BeautifulSoup
import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

TIMEOUT = 2
DX_DOI_URL = "https://dx.doi.org/"
GRID_FILES = ["grid", "full_tables/links",
              "full_tables/addresses", "full_tables/types"]
EU_CODES = ['AT', 'BE', 'HR', 'BG', 'CY', 'CZ', 'DK', 'EE', 'FI',
            'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU',
            'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'GB']
SUBSET_COLS = ["ID", "Name", "Country", "country_code", "lat", "lng", "link"]


def get_latest_grid_doi():
    GRID_URL = "https://www.grid.ac/downloads"
    r = requests.get(GRID_URL)
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", class_="table release")
    row = table.find("tr")
    col = row.find_all("td")[1]
    doi = col.text
    return doi


def get_grid_zipfile_io(doi):
    r = requests.get(DX_DOI_URL+doi)
    soup = BeautifulSoup(r.text, "lxml")
    link = soup.find("a", class_="normal-link download-button shallow-button")
    url = link["href"]
    r = urlopen(url)
    zipfile_io = ZipFile(BytesIO(r.read()))
    return zipfile_io


def build_grid_dataframe(zipfile_io):
    df = None
    for fname in GRID_FILES:
        file_io = zipfile_io.open(fname+".csv")
        _df = pd.read_csv(file_io, low_memory=False)
        if df is None:
            df = _df
            continue
        df = df.join(_df.set_index("grid_id"), on="ID")
    return df


def validate_url(url):
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
    except Exception as exception:
        return str(exception)
    return None


if __name__ == "__main__":
    # Download and build the dataframe
    print("Building dataframe")
    doi = get_latest_grid_doi()
    zipfile_io = get_grid_zipfile_io(doi)
    df = build_grid_dataframe(zipfile_io)
    print("Found", len(df), "institutes")

    # Build and apply a subsetting condition
    condition = df.country_code.apply(lambda x: x in EU_CODES)
    condition = condition & (df["type"] == "Education")
    condition = condition & ~pd.isnull(df.link)
    df = df.loc[condition]
    print("Found", len(df), "EU universities")

    # Validate URLs, and remove erroneous urls
    df["link_error"] = list(map(validate_url, df.link))
    df = df.loc[~pd.isnull(df.link), SUBSET_COLS]
    print("Found", len(df), "EU universities with valid URLs")

    # Sort and save
    df.sort_values(by=["country_code", "Name"], inplace=True)
    df.to_csv("university_urls.csv", index=False)
    print("Saved to disk")
