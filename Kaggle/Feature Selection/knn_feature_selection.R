# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(doMC)

registerDoMC(detectCores())
getDoParWorkers()

tidymodels_prefer()
conflicted::conflicts_prefer(scales::alpha)

# Seed
set.seed(3013)

fv <-  c("x619", 'x105', 'x118', 'x366', 'x652', 'x111', 'x358')
fv <-
  c("x619",
    'x105',
    'x561',
    'x118',
    'x366',
    'x652',
    'x114',
    'x111',
    'x358')

lasso_results<- read_rds("objects/lasso_results.rds")
fv<-tidy(lasso_results) %>% filter(estimate != 0 & term != "(Intercept)") %>% mutate(val = abs(estimate)) %>% select(term,val)
fv<- fv$term
fv <-  c("x619", 'x253', 'x118', 'x653', 'x155')
folds <- vfold_cv(train, v = 5, repeats = 3)
fv <- c(
  'x118',
  'x653',
  'x619',
  'x114',
  'x155',
  'x567',
  'x561',
  'x652',
  'x105',
  'x366',
  'x358'
)


train_all <- read_csv("data/train.csv")
train <- (read_csv("data/train.csv") %>% as_tibble() %>% select(id, all_of(fv), y)) %>% mutate(y = log(y))
test <- read_csv('data/test.csv') %>% as_tibble()%>% select(id,all_of(fv))

recipe <- recipe(y ~ .,
                 train) %>%
  step_rm(id) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_corr(all_predictors())




knn_model <- nearest_neighbor(
  mode = "regression",
  neighbors = tune()
) %>%
  set_engine("kknn")



# set-up tuning grid ----
knn_params <- hardhat::extract_parameter_set_dials(knn_model) %>%
  update(neighbors = neighbors(range = c(1,100)))

# define grid
knn_grid <- grid_regular(knn_params, levels = 15)

# workflow ----
knn_workflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(recipe)

# Tuning/fitting ----
knn_tuned <- knn_workflow %>%
  tune_grid(folds, grid = knn_grid)



knn_workflow_tuned <- knn_workflow %>%
  finalize_workflow(select_best(knn_tuned, metric = "rmse"))

knn_fit_folds <- fit_resamples(
  knn_workflow_tuned, 
  resamples = folds)



total_model_results <- (knn_fit_folds %>% collect_metrics() %>% mutate(model = "knn"))

total_model_results %>% 
  filter(.metric == "rmse") %>% 
  arrange(mean) 



