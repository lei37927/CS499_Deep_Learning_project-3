library(data.table)
library(ggplot2)


source("NNetOneSplit.R") 
source("GradientDescent.R") 

#Use spam data set
spam.dt <- data.table::fread("data/spam.data")

#First scale the input matrix (each column should have mean 0 and variance 1).
label.col.i <- ncol(spam.dt)
X.mat <- as.matrix(spam.dt[, -label.col.i, with=FALSE])
yt.vec <- ifelse(spam.dt[[label.col.i]]==1, 1, -1)
X.sc <- scale(X.mat)

#Next create a variable is.train (logical vector with size equal to the number of observations in the whole data set).
is.train <- sample(rep(c(TRUE, FALSE), length(yt.vec)*c(0.8, 0.2)))
X.train <- X.sc[is.train,]; y.train <- yt.vec[is.train]
X.test <- X.sc[!is.train,]; y.test <- yt.vec[!is.train]

is.subtrain <- sample(rep(c(TRUE, FALSE), length(y.train)*c(0.6, 0.4)))

#GradientDescent (from project 1, logistic regression with number of iterations selected by a held-out validation set)
X.subtrain <- X.train[is.subtrain,]; X.validation <- X.train[!is.subtrain,]
y.subtrain <- y.train[is.subtrain]; y.validation <- y.train[!is.subtrain]

weightMatrix <- GradientDescent(X.subtrain,y.subtrain,stepSize = 0.05,maxIterations = 500)
h.validation <-  1/(1+exp(- as.matrix(X.validation) %*% weightMatrix))
err.h <- colMeans((h.validation >= 0.5) != y.validation)

m_minimizes <- which.min(err.h)
weightVector <- weightMatrix[,m_minimizes]

pred.train.lr <- ifelse(1/(1+exp(- as.matrix(X.train) %*% weightVector)) >0.5, 1, -1) 
pred.test.lr <- ifelse(1/(1+exp(- as.matrix(X.test) %*% weightVector)) >0.5, 1, -1) 


#NN
max.epochs=500;step.size=0.06;n_hidden.units=20
result <- NNetOneSplit(X.train,y.train,max.epochs=max.epochs, step.size=0.06, 
                       n_hidden.units=20,is.subtrain)

best_epochs <- which.min(result$loss.values[,2])
result.best_epochs <- NNetOneSplit(X.train,y.train,max.epochs=best_epochs, step.size=0.06, 
                                   n_hidden.units=20,rep(TRUE,length(y.train)))

#Finally use the learned V.mat/w.vec to make predictions on the test set.
V.mat <- result.best_epochs$V.mat
w.vec <- result.best_epochs$w.vec

pred.train <- ifelse(w.vec %*% (1/(1+exp(- V.mat %*% t(X.train)))) >0.5, 1, -1) 
pred.test <- ifelse(w.vec %*% (1/(1+exp(- V.mat %*% t(X.test))))>0.5, 1, -1)


#accuracy.values
accuracy.values <- data.frame(
  'neural network' = c(
    mean(pred.train == y.train),
    mean(pred.test == y.test)),
  'logistic regression' = c(
    mean(pred.train.lr == y.train),
    mean(pred.test.lr == y.test)),
    check.names = F,row.names = c('train','test'))



df2 <- data.frame(model=rep(c("neural network", "logistic regression"), each=2),
                  split=rep(c("train", "test"),2),
                  accuracy=c(accuracy.values[,1],accuracy.values[,2]))
head(df2)

ggplot(data=df2, aes(x=model, y=accuracy, fill=split)) +
  geom_bar(stat="identity", position=position_dodge())+
  scale_fill_brewer(palette="Paired")+
  theme_minimal()

