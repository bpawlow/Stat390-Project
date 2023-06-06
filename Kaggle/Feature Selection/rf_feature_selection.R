library(tidyverse)
library(tidymodels)
library(vip)
library(stacks)
library(doMC)
registerDoMC(detectCores())
getDoParWorkers()
tidymodels_prefer()

fv <- c("x619", 'x105', 'x561', 'x118', 'x165', 'x366', 'x652', 'x114', 'x111', 'x358')
train <- (read_csv("data/train.csv") %>% as_tibble() %>% select(id, all_of(fv), y)) %>% mutate(y = log(y))
test <- read_csv('data/test.csv') %>% as_tibble()%>% select(id,all_of(fv))
train_all[duplicated(as.list(train_all))]

recipe <- recipe(y ~ .,
                 train) %>%
  step_rm(id) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors())
# Handle common conflicts
tidymodels_prefer()
conflicted::conflicts_prefer(scales::alpha)


rf_mod <- rand_forest(mode = "regression", trees = 100) %>% 
  set_engine("ranger", importance = "impurity")

rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(recipe)


rf_fit<- rf_workflow %>% 
  fit(train)


rf_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 25)


train_y <- (read_csv("data/train.csv") %>% as_tibble() %>% select(id, all_of(fv), y))

train_all <- (read_csv("data/train.csv"))


ggplot(train_y, aes(y)) + 
  geom_histogram() + 
  theme_minim




ggplot(train_all, aes()) + 
  naniar::geom_miss_point()



skimr::skim_without_charts(train_all)


naniar::gg_miss_var(train_y)


view(train)
