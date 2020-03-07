#--- Coding project 3: neural network for binary classification.------------------------------------------
## a stochastic gradient descent algorithm for a neural network with one hidden layer.
#' NNetOneSplit.
#' 
#' @param X.mat train inputs/feature matrix, n_observations x n_features.
#' @param y.vec train outputs/label vector, n_observations x 1.
#' @param max.epochs int scalar > 1.
#' @param step.size double scalar > 0.
#' @param n.hidden.units int scalar>1, number of hidden units.
#' @param is.subtrain logical vector, size n_observations.
#' 
#' @return loss.values a matrix/data.table/etc which stores the logistic loss with respect to the subtrain/validation set for each epoch.
#' @return V.mat, the weight matrix after having done gradient descent for the specified number of epochs (max.epochs).
#' @return w.vec, the weight vector after having done gradient descent for the specified number of epochs (max.epochs).
#' 

NNetOneSplit <- function(X.mat, y.vec, max.epochs=500, step.size=0.06, 
                         n_hidden.units=20,is.subtrain){
  #X.mat <-  X.train; y.vec <- y.train; max.epochs<-500;step.size <- 0.05
  
  X.subtrain <- X.mat[is.subtrain,]
  X.validation <- X.mat[!is.subtrain,]
  y.subtrain <- y.vec[is.subtrain]
  y.validation <- y.vec[!is.subtrain]
  
  n_features.x <- ncol(X.mat)
  n_hidden.units <- 20
  V.mat <- matrix(rnorm(n_features.x*n_hidden.units,0,0.1),
                  ncol = n_features.x,
                  nrow = n_hidden.units)
  w.vec <- matrix(rnorm(n_hidden.units,0,0.1),
                  nrow = 1,
                  ncol = n_hidden.units)
  loss.values <- matrix(NA,nrow = max.epochs,
                        ncol = 2)
  colnames(loss.values) <- c('loss.train','loss.validation')
  
  for (k in 1:max.epochs){
    
    obs.vec <- sample(seq_along(y.subtrain))
    for(iteration.i in seq_along(obs.vec)){#one pass through the train observations.
      obs.i <- obs.vec[[iteration.i]]
      x <- X.subtrain[obs.i,]
      yt <- y.subtrain[obs.i]
      
      ## Forward propagation.
      h <- 1/(1+exp(- V.mat %*% x))
      y <- w.vec %*% h
      
      
      ## Back propagation.
      grad.y <- (-yt / (1+exp(yt*y))) %*% t(h)
      grad.h <- (t(w.vec) %*% (-yt / (1+exp(yt*y))) * h * (1-h)) %*% t(x)
      
      ## Take a step in the negative gradient direction.
      V.mat <- V.mat - step.size*grad.h
      w.vec <- w.vec - step.size*grad.y
      
      ##end of iteration.
    }##end of epoch.
    
    pred.h <- 1/(1+exp(- V.mat %*% t(X.mat)))
    pred.y <- w.vec %*% pred.h
    loss.vec <- log(1+exp(-y.vec*pred.y))
    loss.values[k,] <- c(mean(loss.vec[is.subtrain]),
                         mean(loss.vec[!is.subtrain]))
    
  }
  return(list(loss.values=loss.values,
         V.mat=V.mat,
         w.vec=w.vec))
}




