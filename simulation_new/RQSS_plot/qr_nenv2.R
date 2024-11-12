# Install and load necessary packages
library(ggplot2)
library(dplyr)
library(boot)
library(purrr)
library(tidyr)
library(quantreg)

# Read and prepare data
df <- read.csv("regret_output.csv", stringsAsFactors = FALSE) %>%
  mutate(
    Distance = as.numeric(Distance),
    Regret = as.numeric(Regret),
    Policy = as.factor(Policy),
    n_env = as.factor(n_env)
  )

# Filter for n_env = 2
df_n2 <- df %>%
  filter(n_env == "2") %>%
  filter(!is.na(Distance) & !is.na(Regret))

# Define a grid of Distance values for prediction
distance_grid <- seq(min(df_n2$Distance), max(df_n2$Distance), length.out = 100)

# Function to fit quantile regression and predict
fit_quantile <- function(data, tau) {
  model <- rq(Regret ~ Distance, data = data, tau = tau)
  predict(model, newdata = data.frame(Distance = distance_grid))
}

# Perform bootstrapping for each Policy
bootstrap_results_n2 <- df_n2 %>%
  group_by(Policy) %>%
  nest() %>%
  mutate(
    # Fit lower quantile regression (2.5%)
    quantile_lower = map(data, ~ fit_quantile(.x, tau = 0.25)),
    
    # Fit upper quantile regression (97.5%)
    quantile_upper = map(data, ~ fit_quantile(.x, tau = 0.75)),
    
    # Combine predictions into a data frame
    quantile_df = map2(quantile_lower, quantile_upper, ~ data.frame(
      Distance = distance_grid,
      ci_lower = .x,
      ci_upper = .y
    ))
  ) %>%
  select(Policy, quantile_df) %>%
  unnest(cols = c(quantile_df))

# Plot the results
ggplot() +
  # Scatter plot of the actual data
  geom_point(data = df_n2, 
             aes(x = Distance, y = Regret, color = Policy), 
             alpha = 0.05) +
  
  # Shaded confidence bands
  geom_ribbon(data = bootstrap_results_n2, 
              aes(x = Distance, ymin = ci_lower, ymax = ci_upper, fill = Policy), 
              alpha = 0.3) +
  
  # Quantile regression lines
  geom_line(data = bootstrap_results_n2, 
            aes(x = Distance, y = ci_lower, color = Policy), 
            linetype = "dashed") +
  geom_line(data = bootstrap_results_n2, 
            aes(x = Distance, y = ci_upper, color = Policy), 
            linetype = "dashed") +
  
  # Labels and titles
  labs(
    title = "Regret vs. Distance with Quantile Regression 50% Confidence Bands by Policy (n_env = 2)",
    x = "Distance",
    y = "Regret",
    color = "Policy",
    fill = "Policy"
  ) +
  
  # Theme for aesthetics
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title = element_text(size = 14),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )
