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
    'x017'
  )



# PCA transformation----
# 
# pca_vars <- (read_csv("data/train.csv") %>% as_tibble() %>% select(-all_of(fv))) 
# 
# pca_vars <- pca_vars[!duplicated(as.list(pca_vars))]
# 
# recipe <- recipe(y ~ .,
#                  pca_vars) %>%
#   step_rm(id) %>%
#   step_impute_mode(all_nominal_predictors()) %>%
#   step_dummy(all_nominal_predictors()) %>%
#   step_nzv(all_numeric_predictors()) %>%
#   step_corr(all_numeric_predictors()) %>%
#   step_impute_knn(all_predictors()) %>%
#   step_normalize(all_numeric_predictors())
# 
# 
# 
# pca_data <- recipe %>% prep() %>% bake(pca_vars)
# 
# 
# train_pca <- pca_data %>% 
#   select(-y) %>%
#   scale() %>% # Standardize the variables
#   prcomp() 
# 
# 
# 
# pca_obj<-as_tibble(train_pca$x)

# Loading data ----
train <- read_csv('data/train2.csv')%>% mutate(x516 = as_factor(x516))
pca_train <- read_csv('data/pca_train2.csv') %>% mutate(x516 = as_factor(x516))
folds_pca <- read_rds( 'model selection/model_objects/folds_pca2.rds')
folds <- read_rds( 'model selection/model_objects/folds2.rds')
ctrl_grid <- read_rds('model selection/model_objects/ctrl_grid.rds')



recipe <- recipe(y ~ .,
                 train) %>%
  step_rm(id) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors())



recipe_pca <- recipe(y ~ .,
                    pca_train) %>%
  step_rm(id) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors())





# Boosted Tree ----

bt_model <- boost_tree(
  mode = "regression",
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost")


bt_params <- hardhat::extract_parameter_set_dials(bt_model) %>%
  update(mtry = mtry(c(1, 30)))

bt_grid <- grid_regular(bt_params, levels = 5)

bt_workflow <- workflow() %>%
  add_model(bt_model) %>%
  add_recipe(recipe)

bt_tuned <- bt_workflow %>%
  tune_grid(resamples = folds,
            grid = bt_grid,
            control = ctrl_grid)

write_rds(bt_tuned, "model selection/model_objects/bt_tuned2.rds")

bt_workflow_tuned <- bt_workflow %>%
  finalize_workflow(select_best(bt_tuned, metric = "rmse"))



bt_fit_folds <- fit_resamples(bt_workflow_tuned,
                              resamples = folds,
                              control = control_grid(save_pred = T))




bt_fit_folds %>%
  collect_predictions() %>%
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y))  %>%
  rmse(truth = y,
       estimate = .pred)




bt_fit_folds %>%
  collect_metrics() %>%
  mutate(model = "Boosted Tree") %>%
  filter(.metric == "rmse")



bt_model_pca <- boost_tree(
  mode = "regression",
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost")


# Boosted Tree PCA ----
bt_params_pca <-
  hardhat::extract_parameter_set_dials(bt_model_pca) %>%
  update(mtry = mtry(c(1, 33)))

bt_grid_pca <- grid_regular(bt_params_pca, levels = 5)

bt_workflow_pca <- workflow() %>%
  add_model(bt_model_pca) %>%
  add_recipe(recipe_pca)

bt_tuned_pca <- bt_workflow_pca %>%
  tune_grid(resamples = folds_pca,
            grid = bt_grid_pca,
            control = ctrl_grid)

write_rds(bt_tuned_pca, "model selection/model_objects/bt_tuned_pca2.rds")


bt_workflow_tuned_pca <- bt_workflow_pca %>%
  finalize_workflow(select_best(bt_tuned_pca, metric = "rmse"))



bt_fit_folds_pca <- fit_resamples(bt_workflow_tuned_pca,
                                  resamples = folds_pca,
                                  control = control_grid(save_pred = T))




bt_fit_folds_pca %>%
  collect_predictions() %>%
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y))  %>%
  rmse(truth = y,
       estimate = .pred)




bt_fit_folds_pca %>%
  collect_metrics() %>%
  mutate(model = "Boosted Tree PCA") %>%
  filter(.metric == "rmse")





# Single Layer Neural Network ----
nn_model <- mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune(),
  mode = 'regression'
)

nn_params <- hardhat::extract_parameter_set_dials(nn_model)

nn_grid <- grid_regular(nn_params, levels = 5)

nn_workflow <- workflow() %>%
  add_model(nn_model) %>%
  add_recipe(recipe)

nn_tuned <- nn_workflow %>%
  tune_grid(resamples = folds,
            grid = nn_grid,
            control = ctrl_grid)


write_rds(nn_tuned, "model selection/model_objects/nn_tuned2.rds")


nn_workflow_tuned <- nn_workflow %>%
  finalize_workflow(select_best(nn_tuned, metric = "rmse"))



nn_fit_folds <- fit_resamples(nn_workflow_tuned,
                              resamples = folds,
                              control = control_grid(save_pred = T))




nn_fit_folds %>%
  collect_predictions() %>%
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y))  %>%
  rmse(truth = y,
       estimate = .pred)




nn_fit_folds %>%
  collect_metrics() %>%
  mutate(model = "Neural Network") %>%
  filter(.metric == "rmse")



# Single Layer Neural Network PCA ----
nn_model_pca <- mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune(),
  mode = 'regression'
)

nn_params_pca <- hardhat::extract_parameter_set_dials(nn_model_pca)

nn_grid_pca <- grid_regular(nn_params_pca, levels = 5)

nn_workflow_pca <- workflow() %>%
  add_model(nn_model_pca) %>%
  add_recipe(recipe_pca)

nn_tuned_pca <- nn_workflow_pca %>%
  tune_grid(resamples = folds_pca,
            grid = nn_grid_pca,
            control = ctrl_grid)

write_rds(nn_tuned_pca, "model selection/model_objects/nn_tuned_pca2.rds")


nn_workflow_tuned_pca <- nn_workflow_pca %>%
  finalize_workflow(select_best(nn_tuned_pca, metric = "rmse"))



nn_fit_folds_pca <- fit_resamples(nn_workflow_tuned_pca,
                                  resamples = folds_pca,
                                  control = control_grid(save_pred = T))




nn_fit_folds_pca %>%
  collect_predictions() %>%
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y))  %>%
  rmse(truth = y,
       estimate = .pred)



nn_fit_folds_pca %>%
  collect_metrics() %>%
  mutate(model = "Neural Network PCA") %>%
  filter(.metric == "rmse")





# Cube Model ---- 
cube_model <- parsnip::cubist_rules(
  committees = tune(),
  neighbors = tune()
)



cube_params <- hardhat::extract_parameter_set_dials(cube_model) %>%
  update(neighbors = neighbors(c(5,40)),
         committees = committees(c(3,40))
  )

cube_grid <- grid_regular(cube_params, levels = 5)

cube_workflow <- workflow() %>%
  add_model(cube_model) %>%
  add_recipe(recipe)

cube_tuned <- cube_workflow %>%
  tune_grid(
    resamples = folds,
    grid = cube_grid,
    control = ctrl_grid)

write_rds(cube_tuned, "model selection/model_objects/cube_tuned2.rds")


cube_workflow_tuned <- cube_workflow %>%
  finalize_workflow(select_best(cube_tuned, metric = "rmse"))



cube_fit_folds <- fit_resamples(
  cube_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
)



cube_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


cube_fit_folds %>% 
  collect_metrics() %>%
  mutate(model = "Cube") %>%
  filter(.metric == "rmse") 



# Cube model PCA ----
cube_model_pca <- parsnip::cubist_rules(
  committees = tune(),
  neighbors = tune()
)



cube_params_pca <- hardhat::extract_parameter_set_dials(cube_model_pca) %>%
  update(neighbors = neighbors(c(5,40)),
         committees = committees(c(3,40))
  )

cube_grid_pca <- grid_regular(cube_params_pca, levels = 5)

cube_workflow_pca <- workflow() %>%
  add_model(cube_model_pca) %>%
  add_recipe(recipe_pca)

cube_tuned_pca <- cube_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = cube_grid_pca,
    control = ctrl_grid)

write_rds(cube_tuned_pca, "model selection/model_objects/cube_tuned_pca2.rds")

cube_workflow_tuned_pca <- cube_workflow_pca %>%
  finalize_workflow(select_best(cube_tuned_pca, metric = "rmse"))



cube_fit_folds_pca <- fit_resamples(
  cube_workflow_tuned_pca, 
  resamples = folds_pca,
  control = control_grid(save_pred = T)
)



cube_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


cube_fit_folds_pca %>%
  collect_metrics() %>%
  mutate(model = "Cube PCA") %>%
  filter(.metric == "rmse")






# svmpoly ----
svmpoly_model <- svm_poly(
  mode = "regression",
  cost = tune(),
  degree = tune(),
) %>%
  set_engine("kernlab")

svmpoly_params <- hardhat::extract_parameter_set_dials(svmpoly_model)

svmpoly_grid <- grid_regular(svmpoly_params, levels = 5)

svmpoly_workflow <- workflow() %>%
  add_model(svmpoly_model) %>%
  add_recipe(recipe)

svmpoly_tuned <- svmpoly_workflow %>%
  tune_grid(
    resamples = folds,
    grid = svmpoly_grid,
    control = ctrl_grid
  )


write_rds(svmpoly_tuned, "model selection/model_objects/svmpoly_tuned2.rds")


svmpoly_workflow_tuned <- svmpoly_workflow %>%
  finalize_workflow(select_best(svmpoly_tuned, metric = "rmse"))






svmpoly_fit_folds <- fit_resamples(
  svmpoly_workflow_tuned, 
  resamples = folds, 
  control = control_grid(save_pred = T))



svmpoly_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )





svmpoly_fit_folds %>% 
  collect_metrics() %>% 
  mutate(model = "svmpoly") %>%
  filter(.metric == "rmse")





# svmpoly PCA ----

svmpoly_model_pca <- svm_poly(
  mode = "regression",
  cost = tune(),
  degree = tune(),
) %>%
  set_engine("kernlab")

svmpoly_params_pca <- hardhat::extract_parameter_set_dials(svmpoly_model_pca)

svmpoly_grid_pca <- grid_regular(svmpoly_params_pca, levels = 5)

svmpoly_workflow_pca <- workflow() %>%
  add_model(svmpoly_model_pca) %>%
  add_recipe(recipe_pca)

svmpoly_tuned_pca <- svmpoly_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = svmpoly_grid_pca,
    control = ctrl_grid
  )


write_rds(svmpoly_tuned_pca, "model selection/model_objects/svmpoly_tuned_pca2.rds")


svmpoly_workflow_tuned_pca <- svmpoly_workflow_pca %>%
  finalize_workflow(select_best(svmpoly_tuned, metric = "rmse"))






svmpoly_fit_folds_pca <- fit_resamples(
  svmpoly_workflow_tuned_pca, 
  resamples = folds_pca, 
  control = control_grid(save_pred = T))



svmpoly_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


svmpoly_fit_folds_pca %>% 
  collect_metrics() %>% 
  mutate(model = "svmpoly") %>%
  filter(.metric == "rmse") 



# bmlp ----
bmlp_model <- bag_mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune(),
  mode = 'regression'
)

bmlp_params <- hardhat::extract_parameter_set_dials(bmlp_model)

bmlp_grid <- grid_regular(bmlp_params, levels = 5) 

bmlp_workflow <- workflow() %>%
  add_model(bmlp_model) %>%
  add_recipe(recipe)

bmlp_tuned <- bmlp_workflow %>%
  tune_grid(
    resamples = folds,
    grid = bmlp_grid,
    control = ctrl_grid
  )


write_rds(bmlp_tuned, "model selection/model_objects/bmlp_tuned2.rds")


bmlp_workflow_tuned <- bmlp_workflow %>%
  finalize_workflow(select_best(bmlp_tuned, metric = "rmse"))



bmlp_fit_folds <- fit_resamples(
  bmlp_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
)




bmlp_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


bmlp_fit_folds %>%
  collect_metrics() %>%
  mutate(model = "Bagged Neural Network") %>% 
  filter(.metric == "rmse")





# bmlp PCA ----
bmlp_model_pca <- bag_mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune(),
  mode = 'regression'
)

bmlp_params_pca <- hardhat::extract_parameter_set_dials(bmlp_model_pca)

bmlp_grid_pca <- grid_regular(bmlp_params_pca, levels = 5) 

bmlp_workflow_pca <- workflow() %>%
  add_model(bmlp_model_pca) %>%
  add_recipe(recipe_pca)

bmlp_tuned_pca <- bmlp_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = bmlp_grid_pca,
    control = ctrl_grid
  )


write_rds(bmlp_tuned_pca, "model selection/model_objects/bmlp_tuned_pca2.rds")


bmlp_workflow_tuned_pca <- bmlp_workflow_pca %>%
  finalize_workflow(select_best(bmlp_tuned_pca, metric = "rmse"))



bmlp_fit_folds_pca <- fit_resamples(
  bmlp_workflow_tuned_pca, 
  resamples = folds_pca,
  control = control_grid(save_pred = T)
)




bmlp_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


bmlp_fit_folds_pca %>%
  collect_metrics() %>%
  mutate(model = "bmlp") %>% 
  filter(.metric == "rmse")




#svm rbf ----
svm_model <- svm_rbf(
  mode = "regression",
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab")

svm_params <- hardhat::extract_parameter_set_dials(svm_model)

svm_grid <- grid_regular(svm_params, levels = 5)

svm_workflow <- workflow() %>%
  add_model(svm_model) %>%
  add_recipe(recipe)

svm_tuned <- svm_workflow %>%
  tune_grid(
    resamples = folds,
    grid = svm_grid, 
    control = ctrl_grid
  )

write_rds(svm_tuned, "model selection/model_objects/svm_tuned2.rds")


svm_workflow_tuned <- svm_workflow %>%
  finalize_workflow(select_best(svm_tuned, metric = "rmse"))


svm_fit_folds <- fit_resamples(
  svm_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
)

svm_fit_folds %>%
  collect_predictions() %>%
  mutate(.pred = exp(.pred),
         y = exp(y)) %>%
  rmse(truth = y, estimate = .pred)

svm_fit_folds %>%
  collect_metrics() %>%
  mutate(model = "SVM RBF") %>% 
  filter(.metric == "rmse") 



# svm rbf PCA ----
svm_model_pca <- svm_rbf(
  mode = "regression",
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab")

svm_params_pca <- hardhat::extract_parameter_set_dials(svm_model_pca)

svm_grid_pca <- grid_regular(svm_params_pca, levels = 5)

svm_workflow_pca <- workflow() %>%
  add_model(svm_model_pca) %>%
  add_recipe(recipe_pca)

svm_tuned_pca <- svm_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = svm_grid_pca,
    control = ctrl_grid
  )

write_rds(svm_tuned_pca, "model selection/model_objects/svm_tuned_pca2.rds")


svm_workflow_tuned_pca <- svm_workflow_pca %>%
  finalize_workflow(select_best(svm_tuned_pca, metric = "rmse"))


svm_fit_folds_pca <- fit_resamples(
  svm_workflow_tuned_pca, 
  resamples = folds_pca,
  control = control_grid(save_pred = T)
)

svm_fit_folds_pca %>%
  collect_predictions() %>%
  mutate(.pred = exp(.pred),
         y = exp(y)) %>%
  rmse(truth = y, estimate = .pred)

svm_fit_folds_pca %>% 
  collect_metrics() %>% 
  mutate(model = "SVM RBF PCA") %>% 
  filter(.metric == "rmse") 




# knn ----
knn_model <- nearest_neighbor(
  mode = "regression",
  neighbors = tune()
) %>%
  set_engine("kknn")



knn_params <- hardhat::extract_parameter_set_dials(knn_model) %>%
  update(neighbors = neighbors(range = c(1,100)))

knn_grid <- grid_regular(knn_params, levels = 15)

knn_workflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(recipe)

knn_tuned <- knn_workflow %>%
  tune_grid(folds, 
            grid = knn_grid,
            control = ctrl_grid
  )

write_rds(knn_tuned, "model selection/model_objects/knn_tuned2.rds")


knn_workflow_tuned <- knn_workflow %>%
  finalize_workflow(select_best(knn_tuned, metric = "rmse"))



knn_fit_folds <- fit_resamples(
  knn_workflow_tuned, 
  resamples = folds, 
  control = control_grid(save_pred = T))



knn_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


knn_fit_folds %>% 
  collect_metrics() %>% 
  mutate(model = "knn") %>% 
  filter(.metric == "rmse")




# knn PCA ----
knn_model_pca <- nearest_neighbor(
  mode = "regression",
  neighbors = tune()
) %>%
  set_engine("kknn")



knn_params_pca <- hardhat::extract_parameter_set_dials(knn_model_pca) %>%
  update(neighbors = neighbors(range = c(1,100)))

knn_grid_pca <- grid_regular(knn_params_pca, levels = 15)

knn_workflow_pca <- workflow() %>%
  add_model(knn_model_pca) %>%
  add_recipe(recipe_pca)

knn_tuned_pca <- knn_workflow_pca %>%
  tune_grid(folds_pca, 
            grid = knn_grid_pca,
            control = ctrl_grid
  )

write_rds(knn_tuned_pca, "model selection/model_objects/knn_tuned_pca2.rds")



knn_workflow_tuned_pca <- knn_workflow_pca %>%
  finalize_workflow(select_best(knn_tuned_pca, metric = "rmse"))



knn_fit_folds_pca <- fit_resamples(
  knn_workflow_tuned_pca, 
  resamples = folds_pca, 
  control = control_grid(save_pred = T))



knn_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )




knn_fit_folds_pca %>%
  collect_metrics() %>%
  mutate(model = "knn PCA")%>% 
  filter(.metric == "rmse")






# rf ----

rf_model <- rand_forest(
  mode = "regression",
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger")



rf_params <- hardhat::extract_parameter_set_dials(rf_model)%>% 
  update(mtry = mtry(c(1, 22))) 

rf_grid <- grid_regular(rf_params, levels = 5)

rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe)

rf_tuned <- rf_workflow %>%
  tune_grid(
    resamples = folds,
    grid = rf_grid,
    control = ctrl_grid
  )
write_rds(rf_tuned, "model selection/model_objects/rf_tuned2.rds")


rf_workflow_tuned <- rf_workflow %>%
  finalize_workflow(select_best(rf_tuned, metric = "rmse"))


rf_fit_folds <- fit_resamples(
  rf_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
)



rf_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )



rf_fit_folds %>%
  collect_metrics() %>% 
  mutate(model = "Random Forests") %>%
  filter(.metric == "rmse")




# rf PCA ----
rf_model_pca <- rand_forest(
  mode = "regression",
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger")



rf_params_pca <- hardhat::extract_parameter_set_dials(rf_model_pca)%>% 
  update(mtry = mtry(c(1, 22))) 

rf_grid_pca <- grid_regular(rf_params_pca, levels = 5)

rf_workflow_pca <- workflow() %>%
  add_model(rf_model_pca) %>%
  add_recipe(recipe_pca)

rf_tuned_pca <- rf_workflow %>%
  tune_grid(
    resamples = folds_pca,
    grid = rf_grid_pca,
    control = ctrl_grid
  )

write_rds(rf_tuned_pca, 'model selection/model_objects/rf_tuned_pca2.rds')

rf_workflow_tuned_pca <- rf_workflow_pca %>%
  finalize_workflow(select_best(rf_tuned_pca, metric = "rmse"))


rf_fit_folds_pca <- fit_resamples(
  rf_workflow_tuned_pca, 
  resamples = folds_pca,
  control = control_grid(save_pred = T)
)



rf_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )



rf_fit_folds_pca %>%
  collect_metrics() %>%
  mutate(model = "Random Forests") %>%
  filter(.metric == "rmse")



# Elastic Net ----
en_model <- linear_reg(
  mixture = tune(),
  penalty = tune(),
  mode = "regression",
) %>%
  set_engine("glmnet")

en_params <- hardhat::extract_parameter_set_dials(en_model)

en_grid <- grid_regular(en_params, levels = 5)

en_workflow <- workflow() %>%
  add_model(en_model) %>%
  add_recipe(recipe)

en_tuned <- en_workflow %>%
  tune_grid(
    resamples = folds,
    grid = en_grid,
    control = ctrl_grid)

write_rds(en_tuned, "model selection/model_objects/en_tuned2.rds")



en_workflow_tuned <- en_workflow %>%
  finalize_workflow(select_best(en_tuned, metric = "rmse"))



en_fit_folds <- fit_resamples(
  en_workflow_tuned, 
  resamples = folds, 
  control = control_grid(save_pred = T))



en_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


en_fit_folds %>% 
  collect_metrics() %>%
  mutate(model = "Elastic Net") %>%
  filter(.metric == "rmse")



# Elastic Net PCA ----
en_model_pca <- linear_reg(
  mixture = tune(),
  penalty = tune(),
  mode = "regression",
) %>%
  set_engine("glmnet")

en_params_pca <- hardhat::extract_parameter_set_dials(en_model_pca)

en_grid_pca <- grid_regular(en_params_pca, levels = 5)

en_workflow_pca <- workflow() %>%
  add_model(en_model_pca) %>%
  add_recipe(recipe_pca)

en_tuned_pca <- en_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = en_grid_pca,
    control = ctrl_grid)


write_rds(en_tuned_pca, "model selection/model_objects/en_tuned_pca2.rds")



en_workflow_tuned_pca <- en_workflow_pca %>%
  finalize_workflow(select_best(en_tuned_pca, metric = "rmse"))



en_fit_folds_pca <- fit_resamples(
  en_workflow_tuned_pca, 
  resamples = folds_pca, 
  control = control_grid(save_pred = T))



en_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )



en_fit_folds_pca %>% 
  collect_metrics() %>%
  mutate(model = "Elastic Net PCA") %>%
  filter(.metric == "rmse")





# Linear Model ----
lm_model <- linear_reg() %>%
  set_engine("lm")

lm_workflow <- workflow() %>%
  add_model(lm_model) %>%
  add_recipe(recipe)


lm_fit_folds <- fit_resamples(
  lm_workflow, 
  resamples = folds, 
  control = control_grid(save_pred = T))



lm_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )

lm_fit_folds %>% 
  collect_metrics() %>% 
  mutate(model = "Linear Model") %>%
  filter(.metric == "rmse")



# Linear Model PCA ----
lm_model_pca <- linear_reg() %>%
  set_engine("lm")

lm_workflow_pca <- workflow() %>%
  add_model(lm_model_pca) %>%
  add_recipe(recipe_pca)


lm_fit_folds_pca <- fit_resamples(
  lm_workflow_pca, 
  resamples = folds_pca, 
  control = control_grid(save_pred = T))



lm_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )

lm_fit_folds_pca %>% 
  collect_metrics() %>% 
  mutate(model = "Linear Model PCA") %>%
  filter(.metric == "rmse")



# Lasso Model ----

lasso_model <- linear_reg(mixture = 1, penalty = .01) %>%
  set_engine("glmnet")

lasso_workflow <- workflow() %>%
  add_model(lasso_model) %>%
  add_recipe(recipe)


lasso_fit_folds <- fit_resamples(
  lasso_workflow, 
  resamples = folds, 
  control = control_grid(save_pred = T))



lasso_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


lasso_fit_folds %>%
  collect_metrics() %>%
  mutate(model = "Lasso") %>%
  filter(.metric == "rmse")




# Lasso Model PCA ----

lasso_model_pca <- linear_reg(mixture = 1, penalty = .01) %>%
  set_engine("glmnet")

lasso_workflow_pca <- workflow() %>%
  add_model(lasso_model_pca) %>%
  add_recipe(recipe_pca)


lasso_fit_folds_pca <- fit_resamples(
  lasso_workflow_pca, 
  resamples = folds_pca, 
  control = control_grid(save_pred = T))



lasso_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


lasso_fit_folds_pca %>%
  collect_metrics() %>%
  mutate(model = "Lasso Model PCA") %>%
  filter(.metric == "rmse")




# Bag Tree ----

bag_tree_model <- bag_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune(),
  mode = 'regression'
)

bag_tree_params <- hardhat::extract_parameter_set_dials(bag_tree_model)

bag_tree_grid <- grid_regular(bag_tree_params, levels = 5) 

bag_tree_workflow <- workflow() %>%
  add_model(bag_tree_model) %>%
  add_recipe(recipe)

bag_tree_tuned <- bag_tree_workflow %>%
  tune_grid(
    resamples = folds,
    grid = bag_tree_grid,
    control = ctrl_grid
  )


write_rds(bag_tree_tuned, "model selection/model_objects/bag_tree_tuned2.rds")


bag_tree_workflow_tuned <- bag_tree_workflow %>%
  finalize_workflow(select_best(bag_tree_tuned, metric = "rmse"))



bag_tree_fit_folds <- fit_resamples(
  bag_tree_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
)




bag_tree_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


bag_tree_fit_folds %>%
  collect_metrics() %>%
  mutate(model = "Bag Tree") %>% 
  filter(.metric == "rmse")



# Bag Tree PCA ----
bag_tree_model_pca <- bag_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune(),
  mode = 'regression'
)

bag_tree_params_pca <- hardhat::extract_parameter_set_dials(bag_tree_model_pca)

bag_tree_grid_pca <- grid_regular(bag_tree_params_pca, levels = 5) 

bag_tree_workflow_pca <- workflow() %>%
  add_model(bag_tree_model_pca) %>%
  add_recipe(recipe_pca)

bag_tree_tuned_pca <- bag_tree_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = bag_tree_grid_pca,
    control = ctrl_grid
  )


write_rds(bag_tree_tuned_pca, "model selection/model_objects/bag_tree_tuned_pca2.rds")



bag_tree_workflow_tuned_pca <- bag_tree_workflow_pca %>%
  finalize_workflow(select_best(bag_tree_tuned_pca, metric = "rmse"))



bag_tree_fit_folds_pca <- fit_resamples(
  bag_tree_workflow_tuned_pca,
  resamples = folds_pca,
  control = control_grid(save_pred = T)
)





bag_tree_fit_folds_pca %>%
  collect_predictions() %>%
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y))  %>%
  rmse(truth = y,
       estimate = .pred)


bag_tree_fit_folds_pca %>%
  collect_metrics() %>%
  mutate(model = "bag tree") %>% 
  filter(.metric == "rmse")


# Bag MARS ----
bag_mars_model <- parsnip::bag_mars(
  num_terms = tune(),
  prod_degree = tune(),
  prune_method = tune(),
  mode = 'regression'
)

bag_mars_params <- hardhat::extract_parameter_set_dials(bag_mars_model)

bag_mars_grid <- grid_regular(bag_mars_params, levels = 5) 

bag_mars_workflow <- workflow() %>%
  add_model(bag_mars_model) %>%
  add_recipe(recipe)

bag_mars_tuned <- bag_mars_workflow %>%
  tune_grid(
    resamples = folds,
    grid = bag_mars_grid,
    control = ctrl_grid
  )


write_rds(bag_mars_tuned, "model selection/model_objects/bag_mars_tuned2.rds")


bag_mars_workflow_tuned <- bag_mars_workflow %>%
  finalize_workflow(select_best(bag_mars_tuned, metric = "rmse"))



bag_mars_fit_folds <- fit_resamples(
  bag_mars_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
)




bag_mars_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


bag_mars_fit_folds %>%
  collect_metrics() %>%
  mutate(model = "bag mars") %>% 
  filter(.metric == "rmse")



# Bag MARS PCA ----
bag_mars_model_pca <- parsnip::bag_mars(
  num_terms = tune(),
  prod_degree = tune(),
  prune_method = tune(),
  mode = 'regression'
)


bag_mars_params_pca <- hardhat::extract_parameter_set_dials(bag_mars_model_pca)

bag_mars_grid_pca <- grid_regular(bag_mars_params_pca, levels = 5) 

bag_mars_workflow_pca <- workflow() %>%
  add_model(bag_mars_model_pca) %>%
  add_recipe(recipe_pca)

bag_mars_tuned_pca <- bag_mars_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = bag_mars_grid_pca,
    control = ctrl_grid
  )


write_rds(bag_mars_tuned_pca, "model selection/model_objects/bag_mars_tuned_pca2.rds")


bag_mars_workflow_tuned_pca <- bag_mars_workflow_pca %>%
  finalize_workflow(select_best(bag_mars_tuned_pca, metric = "rmse"))



bag_mars_fit_folds_pca <- fit_resamples(
  bag_mars_workflow_tuned_pca, 
  resamples = folds_pca,
  control = control_grid(save_pred = T)
)





bag_mars_fit_folds_pca %>%
  collect_predictions() %>%
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y))  %>%
  rmse(truth = y,
       estimate = .pred)


bag_mars_fit_folds_pca %>%
  collect_metrics() %>%
  mutate(model = "bag tree") %>% 
  filter(.metric == "rmse")



# MARS model ----
mars_model <- mars(
  num_terms = tune(),
  prod_degree = tune(),
  prune_method = tune(),
  mode = 'regression'
)

mars_params <- hardhat::extract_parameter_set_dials(mars_model)

mars_grid <- grid_regular(mars_params, levels = 5) 

mars_workflow <- workflow() %>%
  add_model(mars_model) %>%
  add_recipe(recipe)

mars_tuned <- mars_workflow %>%
  tune_grid(
    resamples = folds,
    grid = mars_grid,
    control = ctrl_grid
  )

write_rds(mars_tuned, "model selection/model_objects/mars_tuned2.rds")

mars_workflow_tuned <- mars_workflow %>%
  finalize_workflow(select_best(mars_tuned, metric = "rmse"))


mars_fit_folds <- fit_resamples(
  mars_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
)


mars_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )


mars_fit_folds %>%
  collect_metrics() %>%
  mutate(model = "Mars") %>%  
  filter(.metric == "rmse") %>% 
  arrange(mean) 



# MARS model PCA ----
mars_model_pca <- mars(
  num_terms = tune(),
  prod_degree = tune(),
  prune_method = tune(),
  mode = 'regression'
)

mars_params_pca <- hardhat::extract_parameter_set_dials(mars_model_pca)

mars_grid_pca <- grid_regular(mars_params_pca, levels = 5) 

mars_workflow_pca <- workflow() %>%
  add_model(mars_model_pca) %>%
  add_recipe(recipe_pca)

mars_tuned_pca <- mars_workflow_pca %>%
  tune_grid(
    resamples = folds_pca,
    grid = mars_grid_pca,
    control = ctrl_grid
  )
write_rds(mars_tuned_pca, "model selection/model_objects/mars_tuned_pca2.rds")


mars_workflow_tuned_pca <- mars_workflow_pca %>%
  finalize_workflow(select_best(mars_tuned_pca, metric = "rmse"))


mars_fit_folds_pca <- fit_resamples(
  mars_workflow_tuned_pca, 
  resamples = folds_pca,
  control = control_grid(save_pred = T)
)


mars_fit_folds_pca %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )  %>%
  rmse(truth = y,
       estimate = .pred
  )



mars_fit_folds_pca %>%
  collect_metrics() %>% 
  mutate(model = "Mars PCA") %>%
  filter(.metric == "rmse") %>%
  arrange(mean)





