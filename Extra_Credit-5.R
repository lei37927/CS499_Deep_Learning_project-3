library(data.table)
library(ggplot2)
library(plotROC)
source("NNetOneSplit.R") 
source("GradientDescent.R") 


#Use spam data set
spam.dt <- data.table::fread("data/spam.data")

#First scale the input matrix (each column should have mean 0 and variance 1).
label.col.i <- ncol(spam.dt)
X.mat <- as.matrix(spam.dt[, -label.col.i, with=FALSE])
yt.vec <- ifelse(spam.dt[[label.col.i]]==1, 1, -1)
X.sc <- scale(X.mat)

folds = 4

for (fold in 1:folds) {
  is.train <- sample(rep(c(TRUE, FALSE), length(yt.vec)*c(0.8, 0.2)))
  X.train <- X.sc[is.train,]; y.train <- yt.vec[is.train]
  X.test <- X.sc[!is.train,]; y.test <- yt.vec[!is.train]
  
  is.subtrain <- sample(rep(c(TRUE, FALSE), length(y.train)*c(0.6, 0.4)))
  
  #### NN
  max.epochs=500;step.size=0.06;n_hidden.units=20
  result <- NNetOneSplit(X.train,y.train,max.epochs=max.epochs, step.size=0.06, 
                         n_hidden.units=20,is.subtrain)
  
  best_epochs <- which.min(result$loss.values[,2])
  
  result.best_epochs <- NNetOneSplit(X.train,y.train,max.epochs=best_epochs, step.size=0.06, 
                                     n_hidden.units=20,rep(TRUE,length(y.train)))

  V.mat <- result.best_epochs$V.mat
  w.vec <- result.best_epochs$w.vec
  
  pred.train <- w.vec %*% (1/(1+exp(- V.mat %*% t(X.train))))
  pred.test <- w.vec %*% (1/(1+exp(- V.mat %*% t(X.test))))
  
  
  #### LR
  #GradientDescent (from project 1, logistic regression with number of iterations selected by a held-out validation set)
  X.subtrain <- X.train[is.subtrain,]; X.validation <- X.train[!is.subtrain,]
  y.subtrain <- y.train[is.subtrain]; y.validation <- y.train[!is.subtrain]
  
  weightMatrix <- GradientDescent(X.subtrain,y.subtrain,stepSize = 0.05,maxIterations = 500)
  h.validation <-  1/(1+exp(- as.matrix(X.validation) %*% weightMatrix))
  err.h <- colMeans((h.validation >= 0.5) != y.validation)
  
  m_minimizes <- which.min(err.h)
  weightVector <- weightMatrix[,m_minimizes]
  
  pred.train.lr <- 1/(1+exp(- as.matrix(X.train) %*% weightVector))
  pred.test.lr <- 1/(1+exp(- as.matrix(X.test) %*% weightVector))
  
  if(fold == 1){
    pred.values <- list(data.frame(y = c(as.vector(y.test),as.vector(y.test)), 
                                   M = c(as.vector(pred.test),as.vector(pred.test.lr)),
                                   Z = c(rep('NN.1',length(y.test)),rep('LR.1',length(y.test))))
                        )
    
  }else{
    pred.values <- c(pred.values,list(data.frame(y = c(as.vector(y.test),as.vector(y.test)), 
                                                 M = c(as.vector(pred.test),as.vector(pred.test.lr)),
                                                 Z = c(rep(paste('NN','.',fold,sep = ''),length(y.test)),
                                                       rep(paste('LR','.',fold,sep = ''),length(y.test))))
                                      )
    )
  }
}

p<-ggplot(pred.values[[1]], aes(d = y, m = M, color = Z)) + geom_roc() + style_roc() + 
  xlab("FPR") + ylab("TPR") + 
  scale_x_continuous(expand = c(0, 0)) + scale_y_continuous(expand = c(0, 0)) + 
  ggpubr::theme_classic2() + 
  theme(legend.position="top")

p + geom_roc(data=pred.values[[2]], aes(d = y, m = M, color = Z)) +
  geom_roc(data=pred.values[[3]], aes(d = y, m = M, color = Z)) + 
  geom_roc(data=pred.values[[4]], aes(d = y, m = M, color = Z)) 


