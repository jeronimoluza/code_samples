# Overview

This code sample focuses on downloading and processing NDVI data from NASA archives. The program leverages parallel processing to efficiently convert NDVI raster data into vectorized polygons.

# Getting Started

Create the a new environment:

```bash
make create
```

This will create a new environment with the name of this repository. Install the required packages by doing:

```bash
make install
```

Remove the enviroment by doing:

```bash
make clean
```

# Usage

The `run.py` script helps you download and process NDVI data across time. Change the parameters to run the workflow over specific times of the year, or modify the function to download and process continuous temporal data.
