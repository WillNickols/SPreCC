training_log <- read.csv("training.log", sep="\t", header = TRUE)
plot(loss~iteration, training_log)

rolling_mean <- function(x, n = 5){
  cx <- c(0,cumsum(x))
  rsum <- (cx[(n+1):length(cx)] - cx[1:(length(cx) - n)]) / n
  return(rsum)
}

n = 100
plot(1:length(rolling_mean(training_log$loss, n)), rolling_mean(training_log$loss, n), ylim = c(0,10))

n2 = 1000
training_log <- training_log[n2:nrow(training_log),]

mean(abs(training_log$yhat - training_log$y))/
  mean(abs(training_log$y - mean(training_log$y)))

plot(yhat~y, training_log)
summary(lm(yhat~y, training_log))
plot(w_0~iteration, training_log)
plot(w_1~iteration, training_log)
hist(training_log$n_i)

lw1 <- loess(loss~n_i, training_log)
plot(loss~n_i, training_log)
j <- order(training_log$n_i)
lines(training_log$n_i[j],lw1$fitted[j],col="red",lwd=3)

lw1 <- loess(loss~y, training_log)
plot(loss~y, training_log)
j <- order(training_log$y)
lines(training_log$y[j],lw1$fitted[j],col="red",lwd=3)

lw1 <- loess(loss~yhat, training_log)
plot(loss~yhat, training_log)
j <- order(training_log$yhat)
lines(training_log$yhat[j],lw1$fitted[j],col="red",lwd=3)
