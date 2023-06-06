# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(doMC)
registerDoMC(detectCores())
getDoParWorkers()

# Handle common conflicts
tidymodels_prefer()

# Seed
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
    'x516',
    'x654',
    'x685',
    'x591',
    'x585',
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


train <-
  (read_csv("data/train.csv") %>% as_tibble() %>% select(id, all_of(fv), y)) %>% filter(y <= 25) %>%mutate(y = log(y),
                                                                                        x516 = as_factor(x516))

ggplot(train, aes(y)) + 
  geom_histogram()




folds <- vfold_cv(train, v = 5, repeats = 3)





recipe <- recipe(y ~ .,
                 train) %>%
  step_rm(id) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) 


train_mod <-recipe %>% prep() %>% bake(train)


en_model <- linear_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

en_workflow <- workflow() %>%
  add_model(en_model) %>%
  add_recipe(recipe)



# set-up tuning grid ----
EN_params <- hardhat::extract_parameter_set_dials(en_model)
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





en_fit_folds <- fit_resamples(
  en_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
)


fold_predict_en<- en_fit_folds %>% collect_predictions() %>% select(.pred, y) %>% mutate(.pred = exp(.pred),
                                                                                         y = exp(y)
)

write_csv(fold_predict_en,"~/Documents/STAT 390 (not Github)/kaggle competition/fold_predict_outliers.csv")


total_model_results <- (en_fit_folds %>% collect_metrics() %>% mutate(model = "Elastic Net"))

total_model_results %>% 
  filter(.metric == "rmse") %>% 
  arrange(mean) 


nn_fit <- fit(en_workflow_tuned, train)

nn_test <-
  test %>%
  bind_cols(predict(nn_fit, .))



prediction24 <- (
  cbind(id, nn_test)
) %>% select(value,.pred) %>% rename(y = .pred, id = value) %>% mutate(
  y = exp(y),
  y = y + 1
)


write_csv(prediction24, "~/Documents/STAT 390 (not Github)/kaggle competition/predictions/prediction24.csv")




