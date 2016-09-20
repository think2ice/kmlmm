# 1st deliverable: kernel ridge regression

#author: think_2ice
set.seed(777)
x <- seq(0.1, 100, (100 - 0.1)/(1052-1))
length(x)
a <- 10
b <- 50
c <- 80
f <- 0.5*sin(x - 10)/(x-a) + 0.8*sin(x-b)/(x-b) + 0.3*sin(x-c)/(x-c)
noise <- rnorm(1052, 0, 0.05)
t <- f + noise

# plot of the function
plot(x,t,type="l")

# Build a linear model so as to predict the real data
d <- data.frame(x,t)
linreg.1 <- lm (d)
plot(x,t,type="l")
abline(linreg.1, col="yellow")

## using quadratic polynomial:

linreg.2 <- lm (t ~ x + I(x^2), d)
plot(x,t,type="l")
points(x, predict(linreg.2), col="red", type="l")

## using polynomial grade 3:

linreg.3 <- lm (t ~ poly(x,3), d)
plot(x,t,type="l")
points(x, predict(linreg.3), col="red", type="l")

## using polynomial grade 6:

linreg.6 <- lm (t ~ poly(x,6), d)
plot(x,t,type="l")
points(x, predict(linreg.6), col="red", type="l")

## using polynomial grade 9:

linreg.9 <- lm (t ~ poly(x,9), d)
plot(x,t,type="l")
points(x, predict(linreg.9), col="blue", type="l")

## using polynomial grade 20:

linreg.20 <- lm (t ~ poly(x,20), d)
plot(x,t,type="l")
points(x, predict(linreg.20), col="green", type="l")

## using polynomial grade 27:
linreg.27 <- lm (t ~ poly(x,27), d)
plot(x,t,type="l")
points(x, predict(linreg.27), col="green", type="l")
# This is the best polynomial regression found (and it clearly underfits the data)

# Let's see if we can improve the situation using the RBF kernel: 
N <- length(x)
sigma <- 2
kk <- tcrossprod(x)
dd <- diag(kk)
myRBF.kernel <- exp(sigma*(-matrix(dd,N,N)-t(matrix(dd,N,N))+2*kk))
lambda <- 0.01
ident.N <- diag(rep(1,N))
alphas <- solve(myRBF.kernel + lambda*ident.N)
alphas <- alphas %*% t
plot(x,t,type="l")
lines(x,myRBF.kernel %*% alphas,col="magenta")
# Seems that we have to work on the regularization because now we overfit
# Let's try with a higher value of lambda
lambda <- 3
ident.N <- diag(rep(1,N))
alphas <- solve(myRBF.kernel + lambda*ident.N)
alphas <- alphas %*% t
plot(x,t,type="l")
lines(x,myRBF.kernel %*% alphas,col="magenta")
# Without another strategy (perhaps training data and test, CV or other
# other things that we learn in the ML course) it is difficult to difference 
# between fitting correctly or overfitting just by looking at the plot
