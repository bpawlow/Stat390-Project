library(tidymodels)
library(stacks)
library(tidyverse)
library(psych)
library(doMC)
library(vip)
library(stacks)
registerDoMC(detectCores())
getDoParWorkers()
tidymodels_prefer()

fv <- c("x619", 'x105', 'x561', 'x118', 'x165', 'x366', 'x652', 'x114', 'x111', 'x358')
train <- (read_csv("data/train.csv") %>% as_tibble() %>% select(id, all_of(fv), y)) %>% mutate(y = log(y))
test <- read_csv('data/test.csv') %>% as_tibble()%>% select(id,all_of(fv))

load("objects/recipe.rda")


lasso_model <- linear_reg(mixture = 1, penalty = .01) %>%
  set_engine("glmnet")

lasso_workflow <- workflow() %>%
  add_model(lasso_model) %>%
  add_recipe(recipe)

lasso_fit <- fit(lasso_workflow, train)

tidy(lasso_fit)

fv_new <-  c("x619", 'x105', 'x118', 'x366', 'x652', 'x111', 'x358')
