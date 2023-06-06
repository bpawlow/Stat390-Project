library(tidyverse)
library(tidymodels)
library(doMC)
parallel::detectCores()
registerDoMC(cores = 10)

set.seed(3013)

fv <-
  c(
    'x105',
    'x102',
    'x561',
    'x702',
    'x696',
    'x567',
    'x111',
    'x369',
    'x003',
    'x516',
    'x654',
    'x685',
    'x591',
    'x585',
    'x420',
    "x619", 
    'x118', 
    'x652', 
    'x114', 
    'x358'
  )



train <- (read_csv("data/train.csv") %>% as_tibble() %>% select(id, all_of(fv), y)) %>% mutate(y = log(y),
                                                                                               x516 = as_factor(x516)
)
not_train <- (read_csv("data/train.csv") %>% as_tibble() %>% select(-all_of(fv)) %>% mutate(y = log(y))                                                                                             )
test <- read_csv('data/test.csv') %>% as_tibble()%>% select(id,all_of(fv)) %>% mutate(x516 = as_factor(x516))
folds <- vfold_cv(train, v = 5, repeats = 3)


recipe <- recipe(y ~ .,
                 train) %>%
  step_rm(id) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_mode(x516) %>%
  step_dummy(x516)%>%
  step_impute_knn(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) 



peek <- recipe %>% prep() %>% bake(train)

rf_model <- rand_forest(
  mode = "regression",
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger")



# set-up tuning grid ----
rf_params <- hardhat::extract_parameter_set_dials(rf_model)%>% 
  update(mtry = mtry(c(1, 11))) 

# define grid
rf_grid <- grid_regular(rf_params, levels = 5)

# workflow ----
rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe)

# Tuning/fitting ----
rf_tuned <- rf_workflow %>%
  tune_grid(
    resamples = folds,
    grid = rf_grid)

rf_workflow_tuned <- rf_workflow %>%
  finalize_workflow(select_best(rf_tuned, metric = "rmse"))


rf_fit_folds <- fit_resamples(
  rf_workflow_tuned, 
  resamples = folds)



total_model_results <- (rf_fit_folds %>% collect_metrics() %>% mutate(model = "Random Forests"))

total_model_results %>% 
  filter(.metric == "rmse") %>% 
  arrange(mean) 

