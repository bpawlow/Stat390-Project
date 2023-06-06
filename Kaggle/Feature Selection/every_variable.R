library(tidyverse)
library(tidymodels)
library(doMC)
parallel::detectCores()
registerDoMC(cores = 10)
set.seed(3013)



train_all <- read_csv('data/train.csv') %>% select(-id,-y,-all_of(fv))

value_total <- tibble(var = character(), mean = numeric())

train <- (read_csv("data/train.csv") %>% as_tibble() %>% select(id, all_of(fv), y)) %>% mutate(y = log(y),
                                                                                               x516 = as_factor(x516)
)
test <- read_csv('data/test.csv') %>% as_tibble()%>% select(id,all_of(fv)) %>% mutate(x516 = as_factor(x516))
folds <- vfold_cv(train, v = 5, repeats = 3)

# fv <-
#   c(
#     'x105',
#     'x102',
#     'x561',
#     'x702',
#     'x696',
#     'x567',
#     'x111',
#     'x369',
#     'x003',
#     'x516',
#     'x654',
#     'x685',
#     'x591',
#     'x585',
#     'x420',
#     "x619", 
#     'x118', 
#     'x652', 
#     'x114', 
#     'x358',
#     'x366',
#     'x506',
#     'x532',
#     'x668',
#     'x168',
#     'x487',
#     'x192',
#     'x017'
#   )
# 
# 
# 


fv <- c(
  'x014',
  'x096',
  'x102',
  'x105',
  'x118',
  'x146',
  'x253',
  'x355',
  'x366',
  'x378',
  'x420',
  'x488',
  'x516',
  'x543',
  'x561',
  'x569',
  'x609',
  'x619',
  'x651',
  'x654',
  'x670',
  'x683',
  'x687',
  'x696',
  'x702',
  'x721',
  'x724',
  'x742',
  'x755',
  'x756'
)


fve <- colnames(train_all[!duplicated(as.list(train_all))])


for (i in fv) {

  if(i=='x516') next

  # fv <-
  #   c(
  #     'x105',
  #     'x102',
  #     'x561',
  #     'x696',
  #     'x567',
  #     'x111',
  #     'x369',
  #     'x003',
  #     'x516',
  #     'x685',
  #     'x591',
  #     'x585',
  #     "x619", 
  #     'x118', 
  #     'x652', 
  #     'x358',
  #     'x366',
  #     'x506',
  #     'x532',
  #     'x668',
  #     'x168',
  #     'x487',
  #     'x192',
  #     'x017'
  #   )
  # 
  # 
  # fv <- c(
  #   'x567',
  #   'x591',
  #   'x668',
  #   'x369',
  #   'x532',
  #   'x366',
  #   'x017',
  #   'x102',
  #   'x487',
  #   'x105',
  #   'x696',
  #   'x652',
  #   'x685',
  #   'x118',
  #   'x506',
  #   'x168',
  #   'x192',
  #   'x516',
  #   'x561',
  #   'x358',
  #   'x619'
  #   
  #   )
  # 
  
  fv <- c('x116',
          'x253',
          'x756',
          'x092',
          'x355',
          'x603',
          'x146',
          'x619',
          'x567',
          'x561',
          'x337',
          'x105',
          'x702',
          'x724',
          'x755',
          'x118',
          'x014',
          'x670',
          'x102',
          'x735',
          'x516',
          'x420')
  
  
fv <- fv[fv != i]


train <- (read_csv("data/train.csv") %>% as_tibble() %>% select(id, all_of(fv), y)) %>% mutate(y = log(y),
                                                                                               x516 = as_factor(x516)
)

#folds <- vfold_cv(train, v = 5, repeats = 3)


recipe <- recipe(y ~ .,
                 train) %>%
  step_rm(id) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors())%>%
  step_impute_mean(all_numeric_predictors()) %>%
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


en_fit_folds <- fit_resamples(
  en_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
)

total_model_results<- en_fit_folds %>%
  collect_predictions() %>%
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y))  %>%
  rmse(truth = y,
       estimate = .pred) %>%
  mutate(var = i) %>%
  select(var, .estimate)



print(total_model_results)


value_total <- rbind(value_total,total_model_results)
print(value_total)

}



