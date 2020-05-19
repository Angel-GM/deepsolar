
library(dplyr)
library(glmnet)
library(randomForest)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(cowplot)


solar = read.csv('deepsolar_ny.csv' )

set.seed(0)
# Excluding targer var and other nonfeature vars.
X = solar %>% select(-solar_system_count,-county, -state) %>% data.matrix()
y = solar$solar_system_count %>% as.vector()

# Imputting n/a with means
for(i in 1:ncol(X)){
  X[is.na(X[,i]), i] <- mean(X[,i], na.rm = TRUE)
}

mu       =   as.vector(apply(X, 2, 'mean'))
sd       =   as.vector(apply(X, 2, 'sd'))
X.orig   =   X





for (i in c(1:n)){
  X[i,]   =    (X[i,] - mu)/sd
}

X = X[,-1]
p = dim(X)[2]
n = dim(X)[1]


# test train split, 80%/20%
n.train = floor(.8*n) 
n.test = n-n.train

M              =     100
Rsq.test.rf    =     rep(0,M)  # rf= randomForest
Rsq.train.rf   =     rep(0,M)
Rsq.test.en    =     rep(0,M)  #en = elastic net
Rsq.train.en   =     rep(0,M)
Rsq.test.ls    =     rep(0,M)  #ls = lasso
Rsq.train.ls   =     rep(0,M)
Rsq.test.rg    =     rep(0,M)  #rg = ridge
Rsq.train.rg   =     rep(0,M)





for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  # fit elastic-net and calculate and record the train and test R squares 
  cv.fit.en           =     cv.glmnet(X.train, y.train, alpha = .5, nfolds = 10)
  fit              =     glmnet(X.train, y.train, alpha = .5, lambda = cv.fit.en$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") 
  y.test.hat       =     predict(fit, newx = X.test, type = "response") 
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  
  
  
  # fit lasso and calculate and record the train and test R squares 
  cv.fit.ls        =     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  fit              =     glmnet(X.train, y.train, alpha = 1, lambda = cv.fit.ls$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") 
  y.test.hat       =     predict(fit, newx = X.test, type = "response") 
  Rsq.test.ls[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.ls[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  
  
  # fit ridge and calculate and record the train and test R squares 
  cv.fit.rg        =     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  fit              =     glmnet(X.train, y.train, alpha = 0, lambda = cv.fit.rg$lambda.min)
  y.train.hat      =     predict(fit, newx = X.train, type = "response") 
  y.test.hat       =     predict(fit, newx = X.test, type = "response") 
  Rsq.test.rg[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rg[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  
  
  
  # fit RF and calculate and record the train and test R squares 
  rf               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.test.hat       =     predict(rf, X.test)
  y.train.hat      =     predict(rf, X.train)
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  
  # cat(sprintf("m=%3.f| Rsq.test.rf=%.2f,  Rsq.test.en=%.2f| Rsq.train.rf=%.2f,  Rsq.train.en=%.2f| \n", m,  Rsq.test.rf[m], Rsq.test.en[m],  Rsq.train.rf[m], Rsq.train.en[m]))
  
}




# Part b, box plots of rsq
testplot = ggplot(melt(Rsq.test), aes(x=Var2, y=value)) + geom_boxplot() + scale_y_continuous(limits = c(.4,1)) +
  labs(title='Test R^2', x='Method', y="R^2")

trainplot = ggplot(melt(Rsq.train), aes(x=Var2, y=value)) + geom_boxplot() + scale_y_continuous(limits = c(.4,1)) +
  labs(title='Train R^2', x='Method', y="R^2")

grid.arrange(testplot, trainplot , nrow=1)



# Part c, 10fold CV curves
plot(cv.fit.rg, sub = 'Ridge')
plot(cv.fit.en, sub = 'Elastic Net')
plot(cv.fit.ls, sub = 'Lasso')






# Bootstrapping

bootstrapSamples =     100
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)        
beta.ls.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.rg.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)      



for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit bs rf
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])
  # fit bs en
  a                =     0.5 # elastic-net
  cv.fit           =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit$lambda.min)
  beta.en.bs[,m]   =     as.vector(fit$beta)

  # fit bs ls
  b                =     1 # lasso
  cv.fit           =     cv.glmnet(X.bs, y.bs, alpha = b, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = b, lambda = cv.fit$lambda.min)
  beta.ls.bs[,m]   =     as.vector(fit$beta)
  
  # fit bs rg
  c                =     0 # rg
  cv.fit           =     cv.glmnet(X.bs, y.bs, alpha = c, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = c, lambda = cv.fit$lambda.min)  
  beta.rg.bs[,m]   =     as.vector(fit$beta)
  
  
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
}

# calculate bootstrapped standard errors / alternatively you could use qunatiles to find upper and lower bounds
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")
ls.bs.sd    = apply(beta.ls.bs, 1, "sd")
rg.bs.sd    = apply(beta.rg.bs, 1, "sd")

# fit rf to the whole data
rf               =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)

# fit en to the whole data
a=0.5 # elastic-net
cv.fit.en           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.en              =     glmnet(X, y, alpha = a, lambda = cv.fit.en$lambda.min)


# fit ls to the whole data
b=1 # lasso
cv.fit.ls           =     cv.glmnet(X, y, alpha = b, nfolds = 10)
fit.ls              =     glmnet(X, y, alpha = b, lambda = cv.fit.ls$lambda.min)

# fit rg to the whole data
c=0 # ridge
cv.fit.rg           =     cv.glmnet(X, y, alpha = c, nfolds = 10)
fit.rg              =     glmnet(X, y, alpha = c, lambda = cv.fit.rg$lambda.min)



betaS.rf               =     data.frame(names(X[1,]), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")

betaS.en               =     data.frame(names(X[1,]), as.vector(fit.en$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "feature", "value", "err")

betaS.ls               =     data.frame(names(X[1,]), as.vector(fit.ls$beta), 2*ls.bs.sd)
colnames(betaS.ls)     =     c( "feature", "value", "err")


betaS.rg               =     data.frame(names(X[1,]), as.vector(fit.rg$beta), 2*rg.bs.sd)
colnames(betaS.rg)     =     c( "feature", "value", "err")



rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)


enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)


lsPlot =  ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)

rgPlot =  ggplot(betaS.rg, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)



grid.arrange(rfPlot, enPlot, lsPlot, rgPlot, nrow = 4)

# we need to change the order of factor levels by specifying the order explicitly.
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = F)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = F)])
betaS.ls$feature     =  factor(betaS.ls$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = F)])
betaS.rg$feature     =  factor(betaS.rg$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = F)])






rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + 
  scale_x_discrete(name="Feature Number") +  
  coord_flip() +labs(title = 'Random Forest', y = "Importance") + theme( axis.text.y = element_text(size = 6))

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + 
   coord_flip() +labs(title = 'Elastic Net', y ='Importance') + scale_x_discrete(limits=rev(topN)) + theme(axis.title.y=element_blank(),
                                                                                          axis.text.y=element_blank(),
                                                                                          axis.ticks.y=element_blank())


lsPlot =  ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + 
  coord_flip() +labs(title = 'Lasso', y ='Importance') + scale_x_discrete(limits=rev(topN)) + theme(axis.title.y=element_blank(),
                                                                                   axis.text.y=element_blank(),
                                                                                   axis.ticks.y=element_blank())

rgPlot =  ggplot(betaS.rg, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + 
  coord_flip() +labs(title = 'Ridge', y ='Importance') + scale_x_discrete(limits=rev(topN))  + theme(axis.title.y=element_blank(),
                                                                                    axis.text.y=element_blank(),
                                                                                    axis.ticks.y=element_blank())


plot_grid(rfPlot,  rgPlot, enPlot, lsPlot,  align = "h", nrow = 1, rel_widths  = c(.28, .24, .24, .24))






## Residual plots

# fit rf to the whole data
rf               =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)

# fit en to the whole data
a=0.5 # elastic-net
cv.fit.en           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.en              =     glmnet(X, y, alpha = a, lambda = cv.fit.en$lambda.min)


# fit ls to the whole data
b=1 # lasso
cv.fit.ls           =     cv.glmnet(X, y, alpha = b, nfolds = 10)
fit.ls              =     glmnet(X, y, alpha = b, lambda = cv.fit.ls$lambda.min)

# fit rg to the whole data
c=0 # ridge
cv.fit.rg           =     cv.glmnet(X, y, alpha = c, nfolds = 10)
fit.rg              =     glmnet(X, y, alpha = c, lambda = cv.fit.rg$lambda.min)



rf.res = (y - rf$predicted)
rg.res = (y - (X %*% fit.rg$beta))
en.res = (y - (X %*% fit.en$beta))
ls.res = (y - (X %*% fit.ls$beta))


residuals_all = cbind(rf.res, rg.res, en.res, ls.res) %>% as.matrix()
colnames(residuals_all) = c("Random Forest", "Ridge" , "Elastic Net", "Lasso")


# Violin plot of residuals
ggplot(melt(residuals_all), aes(x=Var2, y=value)) + geom_violin() + scale_y_continuous() +
  labs(title='Residuals of All Methods', x='Method', y="Residuals")  + 
  stat_summary(fun.data=mean_sdl,mult=1, geom="crossbar", color="black")


























