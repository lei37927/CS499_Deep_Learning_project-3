library(data.table)
library(ggplot2)
source("NNetOneSplit.R") 

#Use spam data set
spam.dt <- data.table::fread("data/spam.data")

#First scale the input matrix (each column should have mean 0 and variance 1).
label.col.i <- ncol(spam.dt)
X.mat <- as.matrix(spam.dt[, -label.col.i, with=FALSE])
yt.vec <- ifelse(spam.dt[[label.col.i]]==1, 1, -1)
X.sc <- scale(X.mat)

accuracy.values <- matrix(NA,nrow = 4,ncol = 2)
colnames(accuracy.values) <- c('accuracy.train','accuracy.validation')

for (fold in 1:4) {
  is.train <- sample(rep(c(TRUE, FALSE), length(yt.vec)*c(0.8, 0.2)))
  X.train <- X.sc[is.train,]; y.train <- yt.vec[is.train]
  X.test <- X.sc[!is.train,]; y.test <- yt.vec[!is.train]
  
  is.subtrain <- sample(rep(c(TRUE, FALSE), length(y.train)*c(0.6, 0.4)))
  
  max.epochs=500;step.size=0.06;n_hidden.units=20
  
  result <- NNetOneSplit(X.train,y.train,max.epochs=max.epochs, step.size=0.06, 
                         n_hidden.units=20,is.subtrain)
  
  best_epochs <- which.min(result$loss.values[,2])
  is.subtrain <- rep(TRUE,length(y.train))
  result.best_epochs <- NNetOneSplit(X.train,y.train,max.epochs=best_epochs, step.size=0.06, 
                                     n_hidden.units=20,is.subtrain)
  
  V.mat <- result.best_epochs$V.mat
  w.vec <- result.best_epochs$w.vec
  
  pred.train <- ifelse(w.vec %*% (1/(1+exp(- V.mat %*% t(X.train)))) >0.5, 1, -1) 
  pred.test <- ifelse(w.vec %*% (1/(1+exp(- V.mat %*% t(X.test))))>0.5, 1, -1)
  
  
  accuracy.values[fold,] <- c(mean(pred.train == y.train),
                              mean(pred.test == y.test))
}


df2 <- data.frame(split=rep(c("train", "test"), each=4),
                  fold=rep(c("1", "2", "3","4"),2),
                  accuracy=c(accuracy.values[,1],accuracy.values[,2]))
head(df2)

ggplot(data=df2, aes(x=fold, y=accuracy, fill=split)) +
  geom_bar(stat="identity", position=position_dodge())+
  scale_fill_brewer(palette="Paired")+
  theme_minimal()

