#### Analysis
# Stephanie Cunningham

# install.packages("caret", "recipes", "ranger", "pROC", "xgboost)

library(caret)
library(pROC)
library(ranger)
library(recipes)
library(tidyverse)
library(xgboost)

# Set seed
set.seed(10)

# Read in data
dat <- read.csv("data/summary_stats.csv")
dat <- dat[,-c(1,3)]

# Split into training and test sets
row_idx <- sample(seq_len(nrow(dat)), nrow(dat))
training <- dat[row_idx < nrow(dat) * 0.8, ]
testing <- dat[row_idx >= nrow(dat) * 0.8, ]

# Scale & center data
training$behavior <- as.factor(training$behavior)
testing$behavior <- as.factor(testing$behavior)

scaling_recipe <- recipe(behavior ~ ., data = training) |> 
  step_center(where(is.numeric)) |> 
  step_scale(where(is.numeric)) |> 
  prep()

training <- bake(scaling_recipe, training)
testing <- bake(scaling_recipe, testing)
dat$behavior <- as.factor(dat$behavior)

### K-Nearest Neighbors
dat_knn <- knn3(behavior ~ ., training)
confusionMatrix(predict(dat_knn, testing, type="class"), testing$behavior)

auc(multiclass.roc(testing$behavior, predict(dat_knn, testing),
                   plot=TRUE, levels=c("A_FLIGHT","P_FLIGHT","SITTING","STND","WALK")))

# Tuning KNN
k_fold_cv <- function(data, k, n) {
  per_fold <- floor(nrow(data) / k)
  fold_order <- sample(seq_len(nrow(data)), 
                       size = per_fold * k)
  fold_rows <- split(
    fold_order,
    rep(1:k, each = per_fold)
  )
  vapply(
    fold_rows,
    \(fold_idx) {
      fold_test <- data[fold_idx, ]
      fold_train <- data[-fold_idx, ]
      scaling_recipe <- recipe(behavior ~ ., data = fold_train) |> 
        step_center(where(is.numeric)) |> 
        step_scale(where(is.numeric)) |> 
        prep()
      fold_train <- bake(scaling_recipe, fold_train)
      fold_test <- bake(scaling_recipe, fold_test)
      fold_knn <- knn3(behavior ~ ., fold_train, k = n)
      calc_auc(fold_knn, fold_test)
    },
    numeric(1)
  ) |> 
    mean()
}

tuning_grid <- expand.grid(
  n = floor(seq(1, 401, length.out = 50)),
  auc = NA
)
tuning_grid$n <- ifelse(
  tuning_grid$n %% 2 == 0,
  tuning_grid$n + 1,
  tuning_grid$n
)
tuning_grid$n

calc_auc <- function(model, data) {
  multiclass.roc(testing$behavior, predict(model, testing)) |> 
    auc() |> 
    suppressMessages()
}

for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$auc[i] <- k_fold_cv(
    dat,
    5,
    n = tuning_grid$n[i]
  )
}
head(arrange(tuning_grid, -auc))

# Plot tuning results
ggplot(tuning_grid, aes(n, auc)) + 
  geom_line() +  geom_point() + labs(x = "k") + theme_bw()

# Run model
final_knn <- knn3(behavior ~ ., training, k = 9)

# create confusion matrix
confusionMatrix(predict(final_knn, testing, type="class"), testing$behavior)

# plot AUC
auc(multiclass.roc(testing$behavior, predict(final_knn, testing), 
                   plot=TRUE, levels=c("A_FLIGHT","P_FLIGHT","SITTING","STND","WALK")))

### Random Forest
dat_rf <- ranger(behavior ~ ., data=training, num.trees=800, mtry=5, min.node.size=1, replace=FALSE, sample.fraction=1)

confusionMatrix(predictions(predict(dat_rf, testing)), testing$behavior)
pred_rf <- as.numeric(predict(dat_rf, testing[,1:29], type = 'response')$predictions)
auc(multiclass.roc(testing$behavior, pred_rf, 
                   plot=TRUE, levels=c("A_FLIGHT","P_FLIGHT","SITTING","STND","WALK")))

# Tune the random forest
tuning_grid <- expand.grid(
  mtry = 1:10,
  min.node.size = c(1, 3, 5), 
  replace = FALSE,                               
  sample.fraction = c(0.4, 0.7, 1),                       
  auc = NA                                               
)

calc_auc <- function(model, data) {
  rf_pred <- as.numeric(predict(model, testing[,1:29], type = 'response')$predictions)
  multiclass.roc(testing$behavior, rf_pred) |> 
    auc() |> 
    suppressMessages()
}

# Set up cross-validation
k_fold_cv <- function(data, k, ...) {
  per_fold <- floor(nrow(data) / k)
  fold_order <- sample(seq_len(nrow(data)), 
                       size = per_fold * k)
  fold_rows <- split(
    fold_order,
    rep(1:k, each = per_fold)
  )
  vapply(
    fold_rows,
    \(fold_idx) {
      fold_test <- data[fold_idx, ]
      fold_train <- data[-fold_idx, ]
      fold_rf <- ranger(behavior ~ ., fold_train, ...)
      calc_auc(fold_rf, fold_test)
    },
    numeric(1)
  ) |> 
    mean()
}

for (i in seq_len(nrow(tuning_grid))) {
  tuning_grid$auc[i] <- k_fold_cv(
    training, 
    k = 5,
    mtry = tuning_grid$mtry[i],
    min.node.size = tuning_grid$min.node.size[i],
    replace = tuning_grid$replace[i],
    sample.fraction = tuning_grid$sample.fraction[i]
  )
}
head(tuning_grid[order(tuning_grid$auc, decreasing=TRUE), ])

final_rf <- ranger(behavior ~ ., data=training, num.trees=800, mtry=6, min.node.size=5, replace=FALSE, sample.fraction=1)

confusionMatrix(predictions(predict(final_rf, testing)), testing$behavior)
pred_rf <- as.numeric(predict(dat_rf, testing[,1:29], type = 'response')$predictions)
auc(multiclass.roc(testing$behavior, pred_rf, 
                   plot=TRUE, levels=c("A_FLIGHT","P_FLIGHT","SITTING","STND","WALK")))

#### Gradient Boosting Machines

# Set up x and y vectors/matrices
y_train <- as.integer(training$behavior) - 1
y_test <- as.integer(testing$behavior) - 1
x_train <- training %>% select(-behavior)
x_test <- testing %>% select(-behavior)

xgb_train <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)
xgb_test <- xgb.DMatrix(data = as.matrix(x_test), label = y_test)

# Run model
xgb_params <- list(
  booster = "gbtree",
  eta = 0.01,
  max_depth = 8,
  gamma = 4,
  subsample = 0.75,
  colsample_bytree = 1,
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = 5
)

dat_gbm <- xgb.train(
  params = xgb_params,
  data = xgb_train,
  nrounds = 1000,
  early_stopping_rounds=100,
  watchlist=list(train=xgb_train, val=xgb_test)
)

## Tuning 
start.time <- Sys.time()

# empty lists
lowest_error_list = list()
parameters_list = list()

# Create 500 rows with random hyperparamters
for (iter in 1:500){
  param <- list(booster = "gbtree",
                objective = "multi:softprob",
                max_depth = sample(3:10, 1),
                eta = runif(1, .01, .3),
                subsample = runif(1, .7, 1),
                colsample_bytree = runif(1, .6, 1),
                min_child_weight = sample(0:10, 1)
  )
  parameters <- as.data.frame(param)
  parameters_list[[iter]] <- parameters
}

# Create object that contains all randomly created hyperparameters
parameters_df = do.call(rbind, parameters_list)

# Use randomly created parameters to create 10,000 XGBoost-models
for (row in 1:nrow(parameters_df)){
  mdcv <- xgb.train(data=xgb_train,
                    booster = "gbtree",
                    objective = "multi:softprob",
                    max_depth = parameters_df$max_depth[row],
                    eta = parameters_df$eta[row],
                    subsample = parameters_df$subsample[row],
                    colsample_bytree = parameters_df$colsample_bytree[row],
                    min_child_weight = parameters_df$min_child_weight[row],
                    nrounds= 300,
                    num_class=5,
                    eval_metric = "mlogloss",
                    early_stopping_rounds= 30,
                    print_every_n = 100,
                    watchlist = list(train=xgb_train, val=xgb_test)
  )
  lowest_error <- as.data.frame(1 - min(mdcv$evaluation_log$val_mlogloss))
  lowest_error_list[[row]] <- lowest_error
}

# Create object that contains all accuracy's
lowest_error_df = do.call(rbind, lowest_error_list)

# Bind columns of accuracy values and random hyperparameter values
randomsearch = cbind(lowest_error_df, parameters_df)

# Quickly display highest accuracy
max(randomsearch$`1 - min(mdcv$evaluation_log$val_mlogloss)`)

# Stop time and calculate difference
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

write_csv(randomsearch, "data/randomsearch.csv")

# Prepare table
randomsearch <- as.data.frame(randomsearch) %>%
  rename(val_acc = `1 - min(mdcv$evaluation_log$val_mlogloss)`) %>%
  arrange(-val_acc)

params <- list(booster = "gbtree", 
               objective = "multi:softprob",
               max_depth = randomsearch[1,]$max_depth,
               eta = randomsearch[1,]$eta,
               subsample = randomsearch[1,]$subsample,
               colsample_bytree = randomsearch[1,]$colsample_bytree,
               min_child_weight = randomsearch[1,]$min_child_weight)
xgb_tuned <- xgb.train(params = params,
                       data = xgb_train,
                       nrounds=1000,
                       print_every_n = 10,
                       eval_metric = "mlogloss",
                       early_stopping_rounds = 30,
                       num_class=5,
                       watchlist = list(train=xgb_train, val=xgb_test))

gbm_preds <- predict(xgb_tuned, as.matrix(x_test), reshape = TRUE)
gbm_preds <- as.data.frame(gbm_preds)
names(gbm_preds) <- levels(dat$behavior)

gbm_preds$PredictedClass <- apply(gbm_preds, 1, function(y) colnames(gbm_preds)[which.max(y)])
gbm_preds$ActualClass <- levels(dat$behavior)[y_test + 1]

confusionMatrix(factor(gbm_preds$ActualClass), factor(gbm_preds$PredictedClass))

auc(multiclass.roc(testing$behavior, gbm_preds[,1:5], 
                   plot=TRUE, levels=c("A_FLIGHT","P_FLIGHT","SITTING","STND","WALK")))



