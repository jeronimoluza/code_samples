# ============================================================
# get_osm_highways.R
# ============================================================
# This script retrieves highway data from OpenStreetMap (OSM) 
# for a specified country and administrative region.
#
# Features:
# - Queries OSM for highway networks based on a bounding box.
# - Supports filtering by administrative Level 1 regions.
# - Saves the extracted road network data to a CSV file.
# - Generates and saves a plot of the highway network (optional).
#
# Dependencies:
# - osmdata
# - geodata
# - sf
# - ggplot2
# - dplyr
#
# Usage:
# get_osm_highways(country_iso3, level = 0, name_level1 = NULL, plot = FALSE)
#
# Parameters:
# - country_iso3: The ISO3 country code (e.g., "COL" for Colombia).
# - level: Administrative level (default is 0 for the whole country).
# - name_level1: (Optional) Name of the Level 1 region to filter (e.g., "Bogotá D.C.").
# - plot: If TRUE, saves a PNG visualization of the highway network.
#
# Output:
# - CSV file containing highway data (csv/{country_iso3}_highways.csv).
# - PNG visualization if 'plot' is enabled (png/{country_iso3}_highways.png).
#
# Error Handling:
# - If an invalid Level 1 name is provided, the script stops and lists available names.
#
# Example:
# get_osm_highways("COL", level = 1, name_level1 = "Bogotá D.C.", plot = TRUE)
#
# ============================================================

install.packages(c("osmdata","geodata","sf","ggplot2","dplyr"))

library(osmdata)
library(geodata)
library(sf)
library(ggplot2)
library(dplyr)

assign("has_internet_via_proxy", TRUE, environment(curl::has_internet))

get_osm_highways <- function(country_iso3, level = 0, name_level1 = NULL, plot = FALSE) {
  # Create output directories
  dir.create("csv", showWarnings = FALSE)
  dir.create("png", showWarnings = FALSE)

  country_iso3 = toupper(country_iso3)

  # Load GADM data
  gadm_data <- gadm(country = country_iso3, level = level, path = "./data")
  
  # Check if Level 1 name exists
  if (!is.null(name_level1) && !(name_level1 %in% unique(gadm_data$NAME_1))) {
    stop(paste("There is no", name_level1, "at Level 1. Available names are:\n", 
               paste(unique(gadm_data$NAME_1), collapse = ", ")))
  }
  
  # If level1 name is provided, filter the region
  if (!is.null(name_level1)) {
    region <- gadm_data[gadm_data$NAME_1 == name_level1, ]
  } else {
    region <- gadm_data  # Use whole country if no specific region is provided
  }
  
  # Convert region to sf object
  region_sf <- st_as_sf(region)
  
  # Get bounding box for OSM query
  region_bbox <- st_bbox(region_sf)
  
  # Query OSM for highways within the bounding box
  osm_data <- opq(bbox = region_bbox) %>%
    add_osm_feature(key = "highway", 
                    value = c("motorway", "motorway_link", "primary", "primary_link", 
                              "residential", "secondary", "secondary_link", 
                              "tertiary", "tertiary_link", "trunk", "trunk_link", 
                              "unclassified")) %>%
    osmdata_sf()
  
  # Extract highway lines
  osm_lines <- osm_data$osm_lines
  
  # Filter only features that intersect with the selected region
  osm_filtered <- osm_lines[st_intersects(osm_lines, region_sf, sparse = FALSE), ]
  
  # Convert geometry to WKT and compute lengths
  osm_filtered <- osm_filtered %>%
    mutate(wkt_geometry = st_as_text(geometry),
           length_m = st_length(geometry)) %>%
    select(highway, wkt_geometry, length_m)
  
  # Save to CSV
  csv_filename <- paste0("csv/", country_iso3, "_highways.csv")
  write.csv(osm_filtered, csv_filename, row.names = FALSE)
  message("CSV saved to: ", csv_filename)
  
  # Plot and save PNG if requested
  if (plot) {
    plot_filename <- paste0("png/", country_iso3, "_highways.png")

    if (nzchar(name_level1)) {
      title = paste0("OSM Road Network in ", name_level1, ", ", country_iso3)
    } else {
      title = paste0("OSM Road Network in ", country_iso3)
    }
    
    ggplot() +
      geom_sf(data = region_sf, fill = NA, color = "black", size = 0.5) +  # Region boundary
      geom_sf(data = osm_filtered, color = "black", linewidth = 0.1) +  # OSM highways
      labs(title = title,
           x = "Longitude", y = "Latitude") +
      theme_classic()
    ggsave(filename = plot_filename, width = 10, height = 7)
    
    message("Plot saved to: ", plot_filename)
  }
}

# Example usage:
get_osm_highways("ARG", level = 1, name_level1 = "Ciudad de Buenos Aires", plot = TRUE)


