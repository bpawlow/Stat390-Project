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
    'x168'
  )



train <- (read_csv("data/train.csv") %>% as_tibble() %>% select(id, all_of(fv), y)) %>% mutate(y = log(y),
                                                                                               x516 = as_factor(x516)
)
test <- read_csv('data/test.csv') %>% as_tibble()%>% select(id,all_of(fv)) %>% mutate(x516 = as_factor(x516))
folds <- vfold_cv(train, v = 5, repeats = 3)


recipe <- recipe(y ~ .,
                        train) %>%
  step_rm(id) %>%
  step_impute_mode(x516) %>%
  step_dummy(x516) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) 



peek <- recipe %>% prep() %>% bake(train)

nn_model <- mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune(),
  mode = 'regression'
)

# set-up tuning grid ----
nn_params <- hardhat::extract_parameter_set_dials(nn_model)

# define grid
nn_grid <- grid_regular(nn_params, levels = 5) 

# workflow ----
nn_workflow <- workflow() %>%
  add_model(nn_model) %>%
  add_recipe(recipe)

# Tuning/fitting ----
nn_res <- nn_workflow %>%
  tune_grid(
    resamples = folds,
    grid = nn_grid
  )


nn_workflow_tuned <- nn_workflow %>%
  finalize_workflow(select_best(nn_res, metric = "rmse"))




nn_fit_folds <- fit_resamples(
  nn_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
  )

fold_analysis_predictions <- nn_fit_folds %>%
  collect_predictions() %>%
  select(.pred, y) %>% 
  mutate(
    .pred = exp(.pred),
    y = exp(y)
  )
  


write_csv(fold_analysis_predictions, "fold_analysis_predictions.csv")

total_model_results <- (nn_fit_folds %>% collect_metrics() %>% mutate(model = "Neural Network"))

total_model_results %>% 
  filter(.metric == "rmse") %>% 
  arrange(mean) 


nn_fit <- fit(nn_workflow_tuned, train)

nn_test <-
  test %>%
  bind_cols(predict(nn_fit, .))



prediction10 <- (
  cbind(id, nn_test)
) %>% select(value,.pred) %>% rename(y = .pred, id = value) %>% mutate(
  y = exp(y)
)


write_csv(prediction10, "~/Documents/STAT 390 (not Github)/kaggle competition/predictions/prediction10.csv")




