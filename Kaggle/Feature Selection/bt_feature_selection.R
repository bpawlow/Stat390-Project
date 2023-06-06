library(tidyverse)
library(tidymodels)
library(vip)
library(stacks)
library(doMC)
registerDoMC(detectCores())
getDoParWorkers()
tidymodels_prefer()

train_all <- read_csv("data/train.csv") %>%  mutate(y = log(y))


recipe <- recipe(y ~ .,
                 train_all) %>%
  step_rm(id) %>%
  step_nzv(all_predictors()) %>%
  step_corr(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_impute_knn(all_predictors()) 


datax <- recipe %>% prep() %>% bake(train_all)

set.seed(456)
bt_mod <- boost_tree(
  mtry = 4,
  trees = 1000,
  min_n = 10,
  tree_depth = 3,
  learn_rate = 0.1,
  stop_iter = 10,
  mode = "regression",
  engine = "xgboost",
  sample_size = 0.9
) 

bt_workflow <- 
  workflow() %>% 
  add_model(bt_mod) %>% 
  add_recipe(recipe)


bt_fit<- bt_workflow %>% 
  fit(train_all)



xgb.importance(model = bt_fit$fit$fit$fit) %>% xgb.ggplot.importance(
  top_n=25, measure=NULL, rel_to_first = F) 






# Get variable importance
vip <- vip(bt_fit)

# Print variable importance
print(diabetes_vip)