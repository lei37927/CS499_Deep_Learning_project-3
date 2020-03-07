library(data.table)
library(ggplot2)

source("NNetOneSplit.R") # solve method for variance matrices

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

#Next create a variable is.subtrain (logical vector with size equal to the number of observations in the train set).
is.subtrain <- sample(rep(c(TRUE, FALSE), length(y.train)*c(0.6, 0.4)))


#Use NNetOneSplit with the train set as X.mat/y.vec, with is.subtrain as specified above, and a large number for max.epochs.
max.epochs=500;step.size=0.06;n_hidden.units=20

result <- NNetOneSplit(X.train,y.train,max.epochs=max.epochs, step.size=0.06, 
                   n_hidden.units=20,is.subtrain)

#Plot the subtrain/validation loss as a function of the number of epochs, and draw a point to emphasize the minimum of the validation loss curve.
df2 <- data.frame(set=rep(c("subtrain", "validation"),c(max.epochs,max.epochs)),
                  epochs=c(seq(1,max.epochs),seq(1,max.epochs)),
                  loss=c(result$loss.values[,1],result$loss.values[,2]))

 ggplot(df2, aes(x=epochs, y=loss, group=set)) +
  geom_line(aes(color=set)) + 
  theme_minimal() + 
  theme(legend.position="top") + 
  geom_point(size = 1.5, aes(which.min(result$loss.values[,2]), min(result$loss.values[,2])),color="#000000")



#Define a variable called best_epochs which is the number of epochs which minimizes the validation loss.
best_epochs <- which.min(result$loss.values[,2])
is.subtrain <- rep(TRUE,length(y.train))
result.best_epochs <- NNetOneSplit(X.train,y.train,max.epochs=best_epochs, step.size=0.06, 
                       n_hidden.units=20,is.subtrain)

#Finally use the learned V.mat/w.vec to make predictions on the test set.
V.mat <- result.best_epochs$V.mat
w.vec <- result.best_epochs$w.vec

pred.train <- ifelse(w.vec %*% (1/(1+exp(- V.mat %*% t(X.train)))) >0.5, 1, -1) 
pred.test <- ifelse(w.vec %*% (1/(1+exp(- V.mat %*% t(X.test))))>0.5, 1, -1)

baseline.train <- ifelse(mean(y.train) >=0.5, 1,-1)
baseline.test <- ifelse(mean(y.test) >=0.5, 1,-1)

#Make a table of error rates with three rows (train/validation/test sets) and two columns (logistic regression and baseline)
print("------percent correctly predicted labels in the test set-----")
print(data.frame(
  'neural network'=c(
    paste(round(100*mean(pred.train == y.train), 3), "%", sep=""),
    paste(round(100*mean(pred.test == y.test), 3), "%", sep="")),
  'baseline'=c(
    paste(round(100*mean(baseline.train == y.train), 3), "%", sep=""),
    paste(round(100*mean(baseline.test == y.test), 3), "%", sep="")),
  check.names = F,row.names = c('train','test')))
cat("-------------------------------------------------------------------\n\n")


