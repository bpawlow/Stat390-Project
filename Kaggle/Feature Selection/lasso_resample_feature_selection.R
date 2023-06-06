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
folds <- vfold_cv(train, v = 10, repeats = 5)


recipe <- recipe(y ~ .,
                 train) %>%
  step_rm(id) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_mode(x516) %>%
  step_dummy(x516)%>%
  step_impute_knn(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) 



#peek <- recipe %>% prep() %>% bake(train)


lasso_model <- linear_reg(mixture = 1, penalty = .01) %>%
  set_engine("glmnet")

lasso_workflow <- workflow() %>%
  add_model(lasso_model) %>%
  add_recipe(recipe)



lasso_fit_folds <- fit_resamples(
  lasso_workflow, 
  resamples = folds)



total_model_results <- (lasso_fit_folds %>% collect_metrics() %>% mutate(model = "lasso"))

total_model_results %>% 
  filter(.metric == "rmse") %>% 
  arrange(mean) 

