#### Data cleaning and calculation of summary statistics
# Stephanie Cunningham

# install.packages(c("tidyverse","caTools","moments"))
library(tidyverse)
library(caTools)
library(moments)

set.seed(10)

# Read data
dat <- read.csv("data/storks_obs_train.csv")

# Create an index for each data burst
dat$burst <- seq(1,nrow(dat),1)

# Condense data into columns
dat <- dat %>% group_by(burst) %>% 
  gather(key="Axis", value="Value", 1:120) %>%
  arrange(burst) 
dat <- as.data.frame(dat)

# Add an index for acceleration value within burst
dat$index <- rep(seq(1,40,1),length(unique(dat$burst)))

# Remove numbers and punctuation from Axis column
dat$Axis <- gsub("[0-9.]", "", dat$Axis, perl=TRUE)

# spread out axes (one column per axis)
x <- subset(dat, Axis=="X")
y <- subset(dat, Axis=="Y")
z <- subset(dat, Axis=="Z")
Y <- c(y$Value)
Z <- c(z$Value)
dat <- data.frame(x, Y=Y, Z=Z)
dat <- dat[,c(1,2,5,4,6,7)]
names(dat)[4] <- "X"

# Determine unique bursts
bursts <- unique(dat$burst)

# Create empty data frame for summary statistics
sum.stats <- data.frame()

# Calculate summary statistics for each accelerometer burst
for (i in 1:length(bursts)) {
  
  temp <- subset(dat, burst==bursts[i])
  beh <- temp[1,c(1,2)]
  
  # Extract static and dynamic acceleration from raw data
  temp$Static.X <- runmean(temp$X, k=5, alg="fast", endrule="mean", align="center")
  temp$Static.Y <- runmean(temp$Y, k=5, alg="fast", endrule="mean", align="center")
  temp$Static.Z <- runmean(temp$Z, k=5, alg="fast", endrule="mean", align="center")
  temp$Dynamic.X <- temp$X-temp$Static.X
  temp$Dynamic.Y <- temp$Y-temp$Static.Y
  temp$Dynamic.Z <- temp$Z-temp$Static.Z
  
  # Summary Statistics
  odba <- mean(abs(temp$Dynamic.X)+abs(temp$Dynamic.Y)+abs(temp$Dynamic.Z))
  mean.dxy <- mean(temp$Dynamic.X-temp$Dynamic.Y)
  mean.dyz <- mean(temp$Dynamic.Y-temp$Dynamic.Z)
  mean.dxz <- mean(temp$Dynamic.X-temp$Dynamic.Z)
  sd.dxy <- sd(temp$Dynamic.X-temp$Dynamic.Y)
  sd.dyz <- sd(temp$Dynamic.Y-temp$Dynamic.Z)
  sd.dxz <- sd(temp$Dynamic.X-temp$Dynamic.Z)
  sk.dx <- skewness(temp$Dynamic.X)
  ku.dx <- kurtosis(temp$Dynamic.X)
  sk.dy <- skewness(temp$Dynamic.Y)
  ku.dy <- kurtosis(temp$Dynamic.Y)
  sk.dz <- skewness(temp$Dynamic.Z)
  ku.dz <- kurtosis(temp$Dynamic.Z)
  cov.dxy <- cov(temp$Dynamic.X, temp$Dynamic.Y)
  cov.dxz <- cov(temp$Dynamic.X, temp$Dynamic.Z)
  cov.dyz <- cov(temp$Dynamic.Y, temp$Dynamic.Z)
  cor.dxy <- cor(temp$Dynamic.X, temp$Dynamic.Y)
  cor.dxz <- cor(temp$Dynamic.X, temp$Dynamic.Z)
  cor.dyz <- cor(temp$Dynamic.Y, temp$Dynamic.Z)
  max.dx <- max(temp$Dynamic.X)
  min.dx <- min(temp$Dynamic.X)
  amp.dx <- abs(max.dx-min.dx)
  max.dy <- max(temp$Dynamic.Y)
  min.dy <- min(temp$Dynamic.Y)
  amp.dy <- abs(max.dy-min.dy)
  max.dz <- max(temp$Dynamic.Z)
  min.dz <- min(temp$Dynamic.Z)
  amp.dz <- abs(max.dz-min.dz)
  
  features <- data.frame(odba, mean.dxy,mean.dyz,mean.dxz,sd.dxy,sd.dyz,sd.dxz,cov.dxy,cov.dxz,cov.dyz,
                         cor.dxy,cor.dxz,cor.dyz,sk.dx,ku.dx,sk.dy,ku.dy,sk.dz,ku.dz,
                         max.dx,min.dx,amp.dx,max.dy,min.dy,amp.dy,max.dz,min.dz,amp.dz)
  
  features <- cbind(beh, features)

  sum.stats <- rbind(sum.stats, features)
  
}

write.csv(sum.stats, "data/summary_stats.csv")




