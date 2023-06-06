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
    'x358',
    'x366',
    'x506',
    'x532',
    'x668',
    'x168',
    'x487',
    'x192',
    'x017',
    'x556',
    'x047'
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


en_model <- linear_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

en_workflow <- workflow() %>%
  add_model(en_model) %>%
  add_recipe(recipe)



# set-up tuning grid ----
en_params <- hardhat::extract_parameter_set_dials(en_model)
# define grid
en_grid <- grid_regular(en_params, levels = 15)

# workflow ----
en_workflow <- workflow() %>%
  add_model(en_model) %>%
  add_recipe(recipe)

# Tuning/fitting ----
en_tuned <- en_workflow %>%
  tune_grid(folds, grid = en_grid)



en_workflow_tuned <- en_workflow %>%
  finalize_workflow(select_best(en_tuned, metric = "rmse"))





en_fit_folds1 <- fit_resamples(
  en_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
  )

en_fit_folds1 %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  ) %>%
  rmse(truth = y,
       estimate = .pred
  )



fold_predict_en<- en_fit_folds %>% collect_predictions() %>% select(.pred, y) %>% mutate(.pred = exp(.pred),
                                                                                         y = exp(y)
                                                                                         )

write_csv(fold_predict_en,"~/Documents/STAT 390 (not Github)/kaggle competition/fold_predict_en.csv")


total_model_results <- (en_fit_folds %>% collect_metrics() %>% mutate(model = "Elastic Net"))

total_model_results %>% 
  filter(.metric == "rmse") %>% 
  arrange(mean) 





















