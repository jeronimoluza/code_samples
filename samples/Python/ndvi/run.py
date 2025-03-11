import os

import pandas as pd
from shapely import wkt

from src import ndvi


def process_year(year):
    download_dir = "./data/raw"

    # Define search parameters
    short_name = "MOD13Q1"
    version = "061"  # Latest version of MOD13Q1
    start_date = f"{year}-09-14"
    end_date = f"{year}-09-16"
    count = 1

    data_field = "250m 16 days NDVI"

    # Tomé Açu, Pará, Brazil
    # https://www.sciencedirect.com/science/article/pii/S0264837719318447
    region = wkt.loads(
        "POLYGON((-48.837 -1.83, -47.6921 -1.83, -47.6921 -3.3766, -48.837 -3.3766, -48.837 -1.83))"
        # "POLYGON((-59.523568 -34.586749, -59.320925 -34.586749, -59.320925 -34.709499, -59.523568 -34.709499, -59.523568 -34.586749))"
    )

    files = ndvi.download_earthaccess(
        download_dir=download_dir,
        short_name=short_name,
        version=version,
        start_date=start_date,
        end_date=end_date,
        region=region,
        count=count,
    )
    single_file = files[0]
    output_filename = single_file.replace("raw", "extracted").replace(".hdf", ".csv")
    # result = process_file(filepath=single_file, data_field=data_field, geometry=region)
    result = ndvi.process_file(
        filepath=single_file, data_field=data_field, geometry=region
    )
    result.to_csv(output_filename, index=None)


def do_years():
    years = [2009, 2011, 2013, 2015, 2018, 2020]
    for year in years:
        process_year(year)


def post_process_file(f):
    data = pd.read_csv(f)
    julian = f.split("/")[-1].split(".")[1].strip("A")
    date = pd.to_datetime(julian, format="%Y%j")
    data["date"] = date
    return data


def post_process():
    csv_folder = "./data/MOD13Q1_061/extracted"
    files = [
        os.path.join(csv_folder, file)
        for file in os.listdir(csv_folder)
        if file.endswith(".csv")
    ]
    output = []
    for f in files:
        print(f"Postprocessing: {f}")
        data = post_process_file(f)
        output.append(data)
    output = pd.concat(output).reset_index(drop=True)
    output.to_csv("./data/MOD13Q1_061/postprocessed/results.csv", index=None)


if __name__ == "__main__":
    post_process()
