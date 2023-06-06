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


prediction23_analysis<- read_csv("predictions/prediction23.csv")



prediction25_analysis <- read_csv('predictions/prediction25.csv')


outliers <- (prediction23_analysis %>% filter(y >= 3))



prediction30 <- rbind(prediction25_analysis %>% filter(!(id %in% (prediction23_analysis %>% filter(y >= 3))$id)),outliers)




write_csv(prediction30, 'predictions/prediction29.csv')










#---- Double Split Idea resample training
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


train <- (read_csv("data/train.csv") %>% as_tibble() %>% select(id, all_of(fv), y))%>% filter(y <= 25) %>% mutate(y = log(y),
                                                                                               x516 = as_factor(x516)
)

train_new <- (
  read_csv("data/train.csv") %>% as_tibble() %>% select(id, all_of(fv), y)
) %>% mutate(y = log(y),
             x516 = as_factor(x516)
)



not_train <-
  (read_csv("data/train.csv") %>% as_tibble() %>% select(id, all_of(fv), y)) %>% filter(y > 25) %>% mutate(y = log(y),
                                                                                                            x516 = as_factor(x516))

set.seed(3013)
folds_train <- vfold_cv(train, v = 5, repeats = 3)
folds_nottrain <- bootstraps(not_train,
                             times = 100,
                             apparent = T)
folds <-  vfold_cv(train_new, v = 5, repeats = 3)


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





en_fit_folds2 <- fit_resamples(
  en_workflow_tuned, 
  resamples = folds,
  control = control_grid(save_pred = T)
)


en_fit_folds2 %>% 
  collect_predictions() %>% 
  select(.pred, y) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
         ) %>% 
  filter(
         .pred <= 3
         ) %>%
  rmse(truth = y,
       estimate = .pred
       )


en_predictions_new <-
  en_fit_folds %>% 
  collect_predictions() %>% 
  select(.pred, y) %>% 
  mutate(.pred = exp(.pred),
         y = exp(y),
         dif = .pred - y
         )


ggplot(en_predictions_new) + 
  geom_point(aes(.pred,y))


total_model_results <- (en_fit_folds %>% collect_metrics() %>% mutate(model = "Elastic Net"))

total_model_results %>% 
  filter(.metric == "rmse") %>% 
  arrange(mean) 



en_fit <-  fit(en_workflow_tuned, train)


en_test <-
  not_train %>%
  bind_cols(predict(en_fit, .)) %>%
  select(.pred, y) %>%
  mutate(
    y = exp(y),
    .pred = exp(.pred)
  ) %>%
  rmse(truth = y, estimate = .pred)



en_predictions <- (
  cbind(id, en_test)
) %>% select(value,.pred) %>% rename(y = .pred, id = value) %>% mutate(
  y = exp(y)
)






#---- fitting outliers
recipe <- recipe(y ~ .,
                 not_train) %>%
  step_rm(id) %>%
  step_nzv(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors()) %>%
  step_impute_mode(x516) %>%
  step_dummy(x516)%>%
  step_impute_knn(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) 



peek <- recipe %>% prep() %>% bake(not_train)


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
  tune_grid(folds_nottrain, grid = en_grid)



en_workflow_tuned <- en_workflow %>%
  finalize_workflow(select_best(en_tuned, metric = "rmse"))





en_fit_folds <- fit_resamples(
  en_workflow_tuned, 
  resamples = folds_nottrain,
  control = control_grid(save_pred = T)
)


view(en_fit_folds %>% collect_predictions() %>% select(.pred, y) %>% mutate(.pred = exp(.pred),
                                                                       y = exp(y),
                                                                       dif = .pred - y
                                                                       ))

en_fit_folds %>% collect_predictions() %>% select(.pred, y) %>% mutate(.pred = exp(.pred),
                                                                       y = exp(y)
) %>% rmse(truth = y, estimate = .pred)


total_model_results <- (en_fit_folds %>% collect_metrics() %>% mutate(model = "Elastic Net"))

total_model_results %>% 
  filter(.metric == "rmse") %>% 
  arrange(mean) 








j<-left_join(prediction23_analysis,prediction25_analysis, by = 'id') %>% rename( y_23 = y.x, y_25 = y.y) %>% mutate(dif = y_23 - y_25) 

view(left_join(j, prediction27))
