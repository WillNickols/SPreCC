training_log <- read.csv("condition_by_kernel/training.log", sep="\t", header = TRUE)
lw1 <- loess(loss~iteration, training_log)
plot(loss~iteration, training_log)
j <- order(training_log$iteration)
lines(training_log$iteration[j],lw1$fitted[j],col="red",lwd=3)

rolling_mean <- function(x, n = 5){
  cx <- c(0,cumsum(x))
  rsum <- (cx[(n+1):length(cx)] - cx[1:(length(cx) - n)]) / n
  return(rsum)
}

n = 100
plot(1:length(rolling_mean(training_log$loss, n)), rolling_mean(training_log$loss, n), ylim = c(0,1))

n2 = 1000
training_log <- training_log[n2:nrow(training_log),]

mean(abs(training_log$mode - training_log$y))/
  mean(abs(training_log$y - mean(training_log$y)))

plot(mode~y, training_log)
summary(lm(mode~y, training_log))
plot(w_0~iteration, training_log)
plot(w_1~iteration, training_log)
plot(c~iteration, training_log)
plot(loss~mode_prob, training_log)
plot(loss~mean_prob, training_log)
plot(loss~lb, training_log)
plot(loss~ub, training_log)
hist(training_log$n_p)

lw1 <- loess(loss~n_p, training_log)
plot(loss~n_p, training_log)
j <- order(training_log$n_p)
lines(training_log$n_p[j],lw1$fitted[j],col="red",lwd=3)

lw1 <- loess(loss~y, training_log)
plot(loss~y, training_log)
j <- order(training_log$y)
lines(training_log$y[j],lw1$fitted[j],col="red",lwd=3)

lw1 <- loess(loss~mode, training_log)
plot(loss~mode, training_log)
j <- order(training_log$mode)
lines(training_log$mode[j],lw1$fitted[j],col="red",lwd=3)

training_log$interval = training_log$ub - training_log$lb
lw1 <- loess(loss~interval, training_log)
plot(loss~interval, training_log)
j <- order(training_log$interval)
lines(training_log$interval[j],lw1$fitted[j],col="red",lwd=3)

training_log$mode_diff = abs(training_log$mode - training_log$y)
lw1 <- loess(mode_diff~interval, training_log)
plot(mode_diff~interval, training_log)
j <- order(training_log$interval)
lines(training_log$interval[j],lw1$fitted[j],col="red",lwd=3)

plot(ub~lb, training_log)
plot(interval~n_p, training_log)
plot(capture~interval, training_log)
hist(training_log$interval)
