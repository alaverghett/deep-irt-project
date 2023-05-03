# Edit working directory here

setwd(getwd())

library(mirt)

# Read in response data
resp = read.csv("Z:/Code/item-response-theory/project/data/train.csv",header=T)
Resp = data.matrix(resp)
head(Resp)
N = nrow(Resp)
n = ncol(Resp)

# Fit the 2PL model
twopl_mod = 'F=1-23'
(twopl_fit =mirt(Resp, twopl_mod, itemtype = '2PL', SE=TRUE))

# View plots and information
plot(twopl_fit) #TCC
plot(twopl_fit, type = 'trace') #ICCs
plot(twopl_fit, type = 'trace', which.items = c(2, 3, 4, 5), facet_items = FALSE)
plot(twopl_fit, type = 'infotrace') #IIFs
plot(twopl_fit, type = 'info') #TIF
plot(twopl_fit, type = 'infoSE') #TIF and SE
areainfo(twopl_fit,c(-3,3)) #Area under TIF

# Estimated parameters (IRT a parameters contain 1.7 scaling factor by default-matches Logit output MPlus)
# Estimated parameters in slope,threshold metric
twopl_params = coef(twopl_fit)
twopl_params
# Estimated parameters in IRT a,b metric
twopl_params = coef(twopl_fit, IRTpars=T, simplify=T)
twopl_items = twopl_params$items
twopl_items
# Write IRT parameters to file (after removing 1.7 scaling factor)
twopl_items[,1] = twopl_items[,1]/1.7
write.csv(twopl_items[,1:2],file="Param30_Est.csv",row.names = FALSE)

# Estimate trait scores
theta_hat = fscores(twopl_fit,method='EAP')
# Enter name of generating theta file
theta = read.csv("Theta1000.csv",header=T)
# Check recovery-correlation
print(cor(theta,theta_hat))
# Write estimated trait scores to file
write.csv(theta_hat,file=paste("Theta1000","_EAP",".csv",sep = ""),row.names = FALSE)


# step 3a
bias_theta = theta_hat - theta
bias_theta

true_twopl = read.csv("Param30.csv",header=T)
true_twopl = data.matrix(true_twopl)
ab_hat = twopl_items[, c(1,2)]

bias_ab = ab_hat - true_twopl
bias_ab

# step 3b
cor(theta_hat, theta)
cor(ab_hat[,c(1)],true_twopl[,c(1)])
cor(ab_hat[,c(2)],true_twopl[,c(2)])


