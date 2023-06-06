library(tidyverse)
library(tidymodels)
library(psych)
library(doMC)
library(VIM)
registerDoMC(detectCores())
getDoParWorkers()


train <- read_csv("data/train_cleaned.csv")
test <- read_csv("data/test.csv")


lasso_recipe <- recipe(y ~ ., train) %>%
  step_log(y) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_corr(all_predictors())


lm_model <- linear_reg() %>%
  set_engine("lm")

lasso_model <- linear_reg(mixture = 1, penalty = .01) %>%
  set_engine("glmnet")

lasso_workflow <- workflow() %>%
  add_model(lasso_model) %>%
  add_recipe(lasso_recipe)

lm_workflow <- workflow() %>%
  add_model(lm_model) %>%
  add_recipe(lasso_recipe)

lasso_results <- fit(lasso_workflow, train)

lm_results <- fit(lm_workflow, train)

view(tidy(lasso_results))
lasso_results_cleaned <- tidy(lasso_results) %>% filter(estimate !=  0)
view(lasso_results_cleaned)

view(tidy(lm_results))
lm_results_cleaned <- tidy(lm_results) %>% mutate(significant = ifelse(p.value <= .05, "*","")) %>% filter(significant == "*")
view(lm_results_cleaned)


lm_results_cleaned %>% filter(lm_results_cleaned$term %in% lasso_results_cleaned$term)


view((lm_results_cleaned %>% left_join(lasso_results_cleaned, by = 'term')) %>% filter(!is.na(estimate.y)) %>% mutate(new_estimate = abs(estimate.y)))
