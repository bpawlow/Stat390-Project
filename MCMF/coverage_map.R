# Load Packages
library(tidyverse)
library(units)
library(sf)
library(tidycensus)
library(patchwork)


# Read Project Data
prog_data <- read_csv("project_data_v4.csv") %>% 
  unique() %>% 
  filter(category_name %in% c("Academic Support", "Computers", "Math", 
                              "Reading & Writing", "Science", "Social Studies", "Managing Money")) %>% 
  filter(!is.na(latitude)) %>% 
  filter(latitude > 41)

# View program categories
table(prog_data$category_name)

# Read in community area shapefiles
comm_areas <- st_read("Comm_Areas/geo_export_1ee39108-7f41-4d12-9f37-78d44d6d3e5b.shp") %>% 
  select("community")

# Filter down unique program locations and turn into points
prog_data_unique <- prog_data %>%
  mutate(longitude = signif(longitude),
         latitude = signif(latitude)) %>% 
  mutate(lon = longitude,
         lat = latitude) %>% 
  st_as_sf(coords = c("lon", "lat"), crs = st_crs(comm_areas)) %>% 
  select(latitude, longitude) %>% 
  unique() %>% 
  st_buffer(500)

# Get outline of Chicago city limits using tidycensus
chicago_outline <- get_acs(geography = "county subdivision", state = "Illinois", county = "Cook", 
                           variables = "B06011_001", geometry = T) %>% 
  filter(grepl("Chicago", NAME)) %>% 
  select(geometry)

# Load in tidycensus variables for reference
census_2020_vars <- load_variables(2020, "pl")

# Get tidycensus data for child population
kid_data <- get_decennial(geography = "block", year = 2020, state = "Illinois", county = "Cook", 
                          variables = c("P1_001N", "P3_001N"), geometry = T, output = "wide") %>% 
  filter(st_intersects(geometry, chicago_outline, sparse = F)) %>% 
  mutate(value = P1_001N - P3_001N)

# Find child population per square mile
kid_dens_data <- kid_data %>% mutate(area = drop_units(st_area(.))) %>% 
  mutate(density = value / area)

# Create map of program locations overlayed with child population to create map of dead zones
dead_zones <- kid_dens_data %>% 
  ggplot() + geom_sf(aes(fill = density), color = NA) + 
  scale_fill_gradient(limits = c(0, .006), 
                      low = "grey",
                      high = "red",
                      na.value = "red") +
  theme_void() +
  geom_sf(data = prog_data_unique, alpha = .2, fill = "purple", color = NA) +
  theme(legend.position = "none") 

# Save dead zones map
ggsave("dead_zones.jpg", dead_zones, height = 28.6, width = 20.16)

# Create map of child population
child_map <- kid_dens_data %>% 
  ggplot() + geom_sf(aes(fill = density), color = NA) + 
  scale_fill_gradient(limits = c(0, .006), 
                      low = "grey",
                      high = "red",
                      na.value = "red") +
  theme_void() +
  theme(legend.position = "none") 

# Save child population map
ggsave("children_map.jpg", child_map, height = 28.6, width = 20.16)





# Get total population data
pop_data <- get_decennial(geography = "block", year = 2020, state = "Illinois", county = "Cook", 
                          variables = c("P1_001N"), geometry = T) %>% 
  filter(st_intersects(geometry, chicago_outline, sparse = F))

# Find population density
pop_dens_data <- pop_data %>% mutate(area = drop_units(st_area(.))) %>% 
  mutate(density = value / area)

# Create population density map
pop_map <- pop_dens_data %>% 
  ggplot() + geom_sf(aes(fill = density), color = NA) + 
  scale_fill_gradient(limits = c(0, .035), 
                      low = "grey",
                      high = "red",
                      na.value = "red") +
  theme_void() +
  theme(legend.position = "none") 


# Save population density map and child population density map together
ggsave("population_map.jpg", pop_map + child_map, height = 28.6, width = 40.32)
