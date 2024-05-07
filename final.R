#Nico Gonnella and Christian Costanza
#5/5/24
#Final Project: Wisconsin Breast Cancer Data Analysis

#explore data
cancer = read.csv("breast-cancer-wisconsin-data.csv", stringsAsFactors = TRUE)
attach(cancer)
View(cancer)

ncol(cancer)
cancer = cancer[,-33]   #was a whole column of NA values

#separate training and testing data
set.seed (1)
train = sample (1:nrow(cancer), nrow(cancer) / 2)
cancer.test = cancer[-train]
other.cancer.test = diagnosis[-train]


############################################################################################
############################################################################################

#data chosen from radiologyinfo.org

data1 = data[,c(2,4,5,6,8,13,14,15,16,18,23,29)]
data2 = data[,c(4,5,6,8,13,14,15,16,18,23,29)]  #take out qualitative column for corrplot

#exploratory data analysis
pairs(data1)

correlation = cor(data2)
correlation
corrplot(correlation , "circle")


############################################################################################
############################################################################################

#multiple linear regression quantitative model

MLR.cancer = cancer[, c(4,5,6,8,13,14,15,16,18,23,29)]
View(MLR.cancer)

mlr.fit = lm(compactness_mean ~., data = MLR.cancer)
summary(mlr.fit)

par(mfrow = c(2,2))
plot(mlr.fit)
plot(mlr.fit2)
coef(mlr.fit)

confint(mlr.fit)

mlr.cancer2 = MLR.cancer[,c(4,10,11)]
head(mlr.cancer2)
mlr.fit2 = lm(compactness_mean ~., data = mlr.cancer2)
coef(mlr.fit2)
summary(mlr.fit2)
confint(mlr.fit2)
predict(mlr.fit2, data.frame(radius_worst = 25.38, concavity_worst = .7119), interval = "prediction")

############################################################################################
############################################################################################

#Classification model 1: validation set approach and best subset selection to minimize the
#error

library(leaps)
#perform best subset selection on training set
regfit.best = regsubsets(diagnosis ~., data = cancer[train, ], nvmax = 31)

test.mat = model.matrix(diagnosis ~., data = cancer[-train, ])

val.errors <- rep(NA, 31)
confused = ifelse(cancer$diagnosis[-train] == "M", 1, ifelse(cancer$diagnosis[-train] == "B", 0, diagnosis))
for (i in 1:31) {
  coefi = coef(regfit.best, id = i)
  pred = test.mat[, names(coefi)] %*% coefi
  val.errors[i] = mean((confused - pred)^2)
}

val.errors
which.min(val.errors)
coef(regfit.best, 2)
#mse:1.061416
summary(regfit.best)

glm.fit = glm(diagnosis ~ radius_worst + concave.points_worst, family = binomial)
summary(glm.fit)

##########################################################################################

#Classification model 2: lasso selection with cross-validation to fine tune the lambda 
#parameter

x <- model.matrix(diagnosis ~ ., data = cancer)
y <- diagnosis 

library(glmnet)

grid = 10^seq(10, -2, length = 100)
y.numeric = ifelse(y == "M", 1, ifelse(y == "B", 0, diagnosis))
lasso.mod = glmnet(x[train, ], y.numeric[train], alpha = 1, lambda = grid)
par(mfrow = c(1,2))
plot(lasso.mod)

cv.lasso = cv.glmnet(x[train,], y.numeric[train], alpha = 1) #alpha=1 for the Lasso
plot(cv.lasso)

bestlam <- cv.lasso$lambda.min
bestlam

lasso.pred <- predict(lasso.mod , s = bestlam , newx = x[-train , ])
mean (( lasso.pred - y.numeric[-train])^2)
#MSE: 0.07432924

lasso.out = glmnet(x, y.numeric, alpha = 1)
lasso.coef = predict(lasso.out , type = "coefficients", s = bestlam)
lasso.coef

############################################################################################
############################################################################################

#Random Forest Model

library(randomForest)
rf.cancer = randomForest(diagnosis ~., data = cancer, subset = train, mtry = 6, 
                         importance = TRUE)
yhat.rf = predict(rf.cancer, newdata = cancer[-train, ])
yhat.rf = ifelse(yhat.rf == "M", 1, ifelse(yhat.rf == "B", 0, diagnosis))
cancer.test = ifelse(cancer.test == "M", 1, ifelse(cancer.test == "B", 0, diagnosis))

mean((yhat.rf - cancer.test)^2)
#MSE = 1.431875
importance(rf.cancer)

varImpPlot(rf.cancer)


############################################################################################
############################################################################################

#Support Vector Classification

library(e1071) 
svmfit = svm(diagnosis ~., data = cancer[train, ] , kernel = "linear", cost = .01, scale = FALSE) 
svmfit$index 
summary(svmfit) 

ypred <- predict(svmfit, cancer.test) 
table(predict = ypred , truth = cancer$diagnosis)
(339+110)/569 = .789

set.seed(1) 
tune.out <- tune(svm , diagnosis ~., data = cancer[train, ] , kernel = "linear", 
                 ranges = list(cost = c(0.01, 0.1, 1, 2.5, 5, 7.5, 10))) 
summary(tune.out) 
bestmod <- tune.out$best.model 
summary(bestmod) 
ypred <- predict(bestmod, cancer.test) 
table(predict = ypred , truth = cancer$diagnosis) 
(352 + 202)/569 = .974

############################################################################################
############################################################################################