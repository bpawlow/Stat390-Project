data <- read_tsv("Data/project_data_v2.csv") %>% janitor::clean_names() %>%
  mutate(category_name = str_remove_all(category_name, "\\.(?!\\d{2,}$)") %>%
           str_replace_all(",", ".") )

file_path <- "Data/project_data_v3.csv"

# Export the data variable to a CSV file
write.csv(data, file = file_path, row.names = FALSE)
