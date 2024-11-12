library(tidyverse)
library(ggplot2)
library(gridExtra)
library(latex2exp)
data <- read.csv("/Users/jishouxuan/output.csv")

plt.data <- data %>%
    mutate(Acceptance.Rate = as.logical(Acceptance.Rate)) %>%
    filter(Sample.Size %in% c(1000,9000,81000)) %>%
    group_by(Policy, n_env,Sample.Size) %>%
    summarise(acc.rate = mean(Acceptance.Rate))

facet_labels <- setNames(
  paste("Sample size:", unique(data$Sample.Size)),
  unique(data$Sample.Size)
)

ggplot(data = as.data.frame(plt.data),mapping=aes(x=n_env, y=acc.rate, color=Policy))+
    geom_point()+
    geom_line()+
    labs(xlab="# training envs", ylab="Acceptance Rate")+
  facet_wrap(~Sample.Size,labeller = as_labeller(facet_labels))+
  scale_color_manual(
    values = c("$\\emptyset$" = "blue", "X1" = "orange", "X2" = "green", "X1,X2" = "red"),
    labels = c(
      TeX("Ø"), 
      TeX("$X_1$"), 
      TeX("$X_1, X_2$"), 
      TeX("$X_2$")))

  