library(tidyverse)
library(tidymodels)
library(doMC)
library(rules)
parallel::detectCores()
registerDoMC(cores = 10)
tidymodels_prefer()
conflicted::conflicts_prefer(scales::alpha)
conflicted::conflicts_prefer(recipes::update)

set.seed(3013)





folds_residual<- vfold_cv(train, v = 5, strata = y)

svm_model_res <- svm_rbf(
  mode = "regression",
  rbf_sigma = .00316,
  cost = 32
    ) %>%
  set_engine("kernlab")

svm_workflow_res <- workflow() %>%
  add_model(svm_model_res) %>%
  add_recipe(recipe)


svm_fit_folds_res <- fit_resamples(
  svm_workflow_res, 
  resamples = folds_residual,
  control = control_grid(save_pred = T)
)

svm_fit_folds_res %>%
  collect_predictions() %>%
  mutate(.pred = exp(.pred),
         y = exp(y)) %>%
  rmse(truth = y, estimate = .pred)

svm_fit_folds_res %>%
  collect_metrics() %>%
  mutate(model = "SVM RBF") %>% 
  filter(.metric == "rmse") 



train_residual <- (svm_fit_folds_res %>%
  collect_predictions() %>% 
  select(.row, .pred,y) %>%
  left_join(train %>% mutate(.row = id + 1))) %>% arrange(.row) %>% select(-.row)
write_csv(train_residual, 'data/train_residual.csv')




train_residual <- read_csv('data/train_residual.csv') %>%
  mutate(
    .pred = exp(.pred) + 1.69,
    y = exp(y),
    residual = .pred - y
    )
set.seed(3013)
folds_residual<- vfold_cv(train_residual, v = 5, repeats = 3, strata = y)


recipe_residual <- recipe(residual ~ .,
                 train_residual) %>%
  step_rm(id,y) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

svm_model_residual <- svm_rbf(
  mode = "regression",
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab")

svm_params_residual <- hardhat::extract_parameter_set_dials(svm_model_residual)

svm_grid_residual <- grid_regular(svm_params_residual, levels = 5)

svm_workflow_residual <- workflow() %>%
  add_model(svm_model_residual) %>%
  add_recipe(recipe_residual)

svm_tuned_residual <- svm_workflow_residual %>%
  tune_grid(
    resamples = folds_residual,
    grid = svm_grid_residual
    )





svm_workflow_tuned_residual <- svm_workflow_residual %>%
  finalize_workflow(select_best(svm_tuned_residual, metric = "rmse"))


svm_fit_folds_residual <- fit_resamples(
  svm_workflow_tuned_residual, 
  resamples = folds_residual,
  control = control_grid(save_pred = T)
)

svm_fit_folds_residual %>%
  collect_predictions() %>%
  rmse(truth = residual, estimate = .pred) 





train_residual_class <- read_csv('data/train_residual.csv') %>%
  mutate(
    .pred = exp(.pred) + 1.69,
    y = exp(y),
    residual = ifelse(.pred - y <= 0, 'under', 'over'),
    x516 = as_factor(x516)
  ) 
set.seed(3013)

folds_residual_class<- vfold_cv(train_residual_class, v = 5, repeats = 3, strata = y)


recipe_residual_class <- recipe(residual ~ .,
                          train_residual_class) %>%
  step_rm(id,y) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

rec_res_mod <-recipe_residual_class %>% prep() %>% bake(train_residual_class)

svm_model_residual <- logistic_reg(
  mode = "classification",
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

svm_params_residual <- hardhat::extract_parameter_set_dials(svm_model_residual)

svm_grid_residual <- grid_regular(svm_params_residual, levels = 5)

svm_workflow_residual <- workflow() %>%
  add_model(svm_model_residual) %>%
  add_recipe(recipe_residual_class)

svm_tuned_residual <- svm_workflow_residual %>%
  tune_grid(
    resamples = folds_residual_class,
    grid = svm_grid_residual
  )





svm_workflow_tuned_residual <- svm_workflow_residual %>%
  finalize_workflow(select_best(svm_tuned_residual, metric = "accuracy"))


svm_fit_folds_residual <- fit_resamples(
  svm_workflow_tuned_residual, 
  resamples = folds_residual_class,
  control = control_grid(save_pred = T)
)

svm_fit_folds_residual %>%
  collect_predictions() %>%
  accuracy(truth = residual, estimate = .pred_class) 

svm_fit_folds_residual %>%
  collect_predictions() %>%
  filter(.pred_under <= .2 | .pred_under >= .8) %>%
  accuracy(truth = residual, estimate = .pred_class) 


test <- read_csv('data/test.csv')

prediction67 <- read_csv("predictions/prediction67.csv") %>% left_join(test) %>% rename(.pred = y) %>% mutate(y = 1:n())


svm_fit <- fit(svm_workflow_tuned_residual,train_residual_class)

svm_predict<- predict(svm_fit,prediction61) cbind()



prediction61 %>%
  bind_cols(predict(svm_fit, ., type = 'prob'))%>% 
  select(id,.pred, .pred_over,.pred_under) %>% write_csv('excel_prediction_analysis/pred61extra.csv')






read_