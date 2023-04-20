library(tidyverse)
library(skimr)
library(tidymodels)


data <- read_tsv("project_data_v2.csv") %>% janitor::clean_names() %>%
  mutate(category_name = str_remove_all(category_name, "\\.(?!\\d{2,}$)") %>% 
           str_replace_all(",", ".") )

skimr::skim_without_charts(data)

chicago_race_income <-
  (read_csv("chicago_race_income.csv") %>% janitor::clean_names())[1:78, 1:8]

chicago_race_income$community_area[chicago_race_income$community_area == "LITTLE VILLAGE"] <-
  "SOUTH LAWNDALE"

chicago_race_income$community_area[chicago_race_income$community_area == "BACK OF THE YARDS"] <-
  "NEW CITY"

cluster_to_ca <- read_csv("cluster_to_community_area.csv")



data <-
  data %>% select(
    id,
    program_name,
    category_name,
    meeting_type,
    program_pays_participants,
    program_has_scholarships,
    program_provides_transportation,
    program_provides_free_food,
    program_price,
    capacity,
    min_age,
    max_age,
    zipcode,
    latitude,
    longitude,
    geographic_cluster_name
  )


skimr::skim_without_charts(data)


unique(data$geographic_cluster_name)


data$geographic_cluster_name[data$geographic_cluster_name == "Back of the Yards"] <-
  "NEW CITY"
data$geographic_cluster_name[data$geographic_cluster_name == "Little Village"] <-
  "SOUTH LAWNDALE"
data$geographic_cluster_name[data$geographic_cluster_name == "Bronzeville/South Lakefront"] <-
  "DOUGLAS"
data$geographic_cluster_name[data$geographic_cluster_name == "GARFIELD PARK"] <-
  "EAST GARFIELD PARK"


unique(data$geographic_cluster_name)


data <-
  left_join(data,
            chicago_race_income,
            by = c("geographic_cluster_name" = "community_area")) %>% filter(meeting_type != "online", !is.na(geographic_cluster_name))




data <- data %>% mutate(median_household_income_bracket = as_factor(
  ifelse(
    median_household_income < 30000,
    "Less than 30,000",
    ifelse(
      median_household_income >= 30000 &
        median_household_income < 50000,
      "30,000 to 49,999",
      ifelse(
        median_household_income >= 50000 &
          median_household_income < 70000,
        "50,000 to 69,999",
        ifelse(
          median_household_income >= 70000 &
            median_household_income < 90000,
          "70,000 to 89,999",
          "Greater than 90,000"
        )
      )
    )
  )
))

ggplot(data, aes(median_household_income_bracket)) + 
  geom_bar(aes(fill = category_name)) + coord_flip()

data %>% group_by(median_household_income_bracket) %>% summarize(n = n(),total_capacity = sum(capacity, na.rm = T))


chicago_race_income <- chicago_race_income %>% mutate(median_household_income_bracket = as_factor(
  ifelse(
    median_household_income < 30000,
    "Less than 30,000",
    ifelse(
      median_household_income >= 30000 &
        median_household_income < 50000,
      "30,000 to 49,999",
      ifelse(
        median_household_income >= 50000 &
          median_household_income < 70000,
        "50,000 to 69,999",
        ifelse(
          median_household_income >= 70000 &
            median_household_income < 90000,
          "70,000 to 89,999",
          "Greater than 90,000"
        )
      )
    )
  )
)) %>% filter(!is.na(community_area))



left_join(
  chicago_race_income %>% group_by(median_household_income_bracket) %>% summarize(pop = sum(total_population, na.rm = T)),
  data %>% group_by(median_household_income_bracket) %>% summarize(n = n(), total_capacity = sum(capacity, na.rm = T))
) %>% mutate(prop_capacity = total_capacity/pop,
             prop_programs = n/pop
             )



ggplot(chicago_race_income, aes(median_household_income_bracket)) + 
  geom_bar()

data <- data %>% mutate(
  asian_prop = round(not_hispanic_or_latino_asian_alone/total_population,3),
  black_prop = round(not_hispanic_or_latino_black_or_african_american_alone/total_population,3),
  hispanic_prop = round(hispanic_or_latino/total_population,3),
  white_prop = round(not_hispanic_or_latino_white_alone/total_population,3),
  other_prop = round(not_hispanic_or_latino_other/total_population,3)
  
)



data <- left_join(data, cluster_to_ca, by = c("geographic_cluster_name" = "community_area"))



data <-
  data %>% mutate(
    simpson_diversity_index = 1 - ((black_prop ^ 2) + (white_prop ^ 2) + (hispanic_prop ^ 2) + (asian_prop ^ 2) + (other_prop^2)),
    gini_diversity_index = 1 - sqrt(((black_prop ^ 2) + (white_prop ^ 2) + (hispanic_prop ^ 2) + (asian_prop ^ 2) + (other_prop^2)))
  )



write_csv(data, "~/Desktop/Stat390-Project/project_data_v4.csv")
