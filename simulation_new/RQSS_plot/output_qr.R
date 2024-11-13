# -------------------------------------------------
# Corrected Script: Quantile Regression (rq)
# for Multiple n_env Values with 2x2 Grid Visualization
# -------------------------------------------------

# ---------------------------
# 1. Install and Load Packages
# ---------------------------

# Define required packages
required_packages <- c("ggplot2", "dplyr", "quantreg", "tidyr", "readr", "patchwork")

# Install any missing packages
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load the packages
library(ggplot2)
library(dplyr)
library(quantreg)
library(tidyr)
library(readr)
library(patchwork)  # For combining plots

# ---------------------------
# 2. Read and Prepare Data
# ---------------------------

# Read the CSV file into a data frame
df <- read_csv("regret_output.csv")

# Convert columns to appropriate data types
df <- df %>%
  mutate(
    Distance = as.numeric(Distance),
    Regret = as.numeric(Regret),
    Policy = as.factor(Policy),
    n_env = as.factor(n_env)
  )

df <- df |> mutate(Regret = Regret + 1e-7)

# ---------------------------
# 3. Filter Data for Relevant n_env Values
# ---------------------------

# Define the n_env values you want to analyze
desired_nenv <- c("2", "4", "6", "8")  # Modify as needed

# Filter data for the desired n_env values
df_filtered <- df %>%
  filter(n_env %in% desired_nenv)

# Remove rows with NA in critical columns
df_filtered <- df_filtered %>%
  filter(!is.na(Distance) & !is.na(Regret) & !is.na(Policy) & !is.na(n_env))

# ---------------------------
# 4. Define a Grid of Distance Values
# ---------------------------

# Define a grid of Distance values for prediction
distance_grid_df <- df_filtered %>%
  group_by(n_env) %>%
  summarise(
    min_distance = min(Distance, na.rm = TRUE),
    max_distance = max(Distance, na.rm = TRUE)
  ) %>%
  ungroup()

# Create a list to store distance grids for each n_env
distance_grids <- list()

for(i in 1:nrow(distance_grid_df)){
  env <- distance_grid_df$n_env[i]
  min_dist <- distance_grid_df$min_distance[i]
  max_dist <- distance_grid_df$max_distance[i]
  
  # Define 100 equally spaced points for each n_env
  distance_grids[[as.character(env)]] <- seq(from = min_dist, to = max_dist, length.out = 100)
}

# ---------------------------
# 5. Fit Quantile Regression Models for Each n_env and Policy
# ---------------------------

# Get the list of unique n_env and Policies
nenvs <- unique(df_filtered$n_env)
policies <- unique(df_filtered$Policy)

# Initialize an empty list to store predictions
quantile_predictions_all <- list()

# Loop over each n_env
for(env in nenvs){
  
  # Subset data for the current n_env
  df_env <- df_filtered %>%
    filter(n_env == env)
  
  # Get unique Policies within this n_env
  policies_env <- unique(df_env$Policy)
  
  # Loop over each Policy within this n_env
  for(p in policies_env){
    
    # Subset data for the current Policy
    data_p <- df_env %>%
      filter(Policy == p)
    
    # Define the grid for this n_env
    distance_grid_env <- distance_grids[[as.character(env)]]
    
    # Fit Quantile Regression model for tau = 0.025 (lower quantile)
    model_lower <- tryCatch({
      rq(log(Regret) ~ Distance, tau = 0.25, data = data_p)
    }, error = function(e){
      message(paste("Error in fitting lower quantile model for Policy:", p, "in n_env:", env, "-", e$message))
      return(NULL)
    })
    
    # Fit Quantile Regression model for tau = 0.975 (upper quantile)
    model_upper <- tryCatch({
      rq(log(Regret) ~ Distance, tau = 0.75, data = data_p)
    }, error = function(e){
      message(paste("Error in fitting upper quantile model for Policy:", p, "in n_env:", env, "-", e$message))
      return(NULL)
    })
    
    # Proceed only if both models are successfully fitted
    if(!is.null(model_lower) & !is.null(model_upper)){
      
      # Create a new data frame for prediction
      new_data <- data.frame(Distance = distance_grid_env)
      
      # Predict lower quantile
      preds_lower <- predict(model_lower, newdata = new_data) |> exp()
      
      # Predict upper quantile
      preds_upper <- predict(model_upper, newdata = new_data) |> exp()
      
      # Combine predictions into a data frame
      preds_df <- data.frame(
        n_env = env,
        Policy = p,
        Distance = distance_grid_env,
        ci_lower = preds_lower,
        ci_upper = preds_upper
      )
      
      # Append to the list
      quantile_predictions_all[[paste(env, p, sep = "_")]] <- preds_df
    } else {
      message(paste("Skipping Policy:", p, "in n_env:", env, "due to model fitting errors."))
    }
  }
}

# Combine all predictions into a single data frame
quantile_bins_all <- bind_rows(quantile_predictions_all)

# ---------------------------
# 6. Visualization: Create Individual Plots for Each n_env
# ---------------------------

# Initialize an empty list to store individual plots
plot_list <- list()

# Loop over each n_env to create individual plots
for(env in nenvs){
  
  # Subset data for the current n_env
  df_env <- df_filtered %>%
    filter(n_env == env)
  
  # Subset predictions for the current n_env
  preds_env <- quantile_bins_all %>%
    filter(n_env == env)
  
  # Create the plot
  p <- ggplot() +
    # Scatter plot of the actual data
    geom_point(data = df_env, 
               aes(x = Distance, y = Regret, color = Policy), 
               alpha = 0.1) +
    
    # Shaded confidence bands
    geom_ribbon(data = preds_env, 
                aes(x = Distance, ymin = ci_lower, ymax = ci_upper, fill = Policy), 
                alpha = 0.2) +
    
    # Quantile regression lines
    geom_line(data = preds_env, 
              aes(x = Distance, y = ci_lower, color = Policy), 
              linetype = "dashed") +
    geom_line(data = preds_env, 
              aes(x = Distance, y = ci_upper, color = Policy), 
              linetype = "dashed") +
    xlim(5,40) + ylim(0, 5)
    # Title for each subplot indicating n_env
    ggtitle(paste("n_env =", env)) +
    
    # Theme adjustments
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      legend.position = "none"  # Remove individual legends
    )
  
  # Append the plot to the list
  plot_list[[as.character(env)]] <- p
}

# ---------------------------
# 7. Combine Plots into a 2x2 Grid with Shared Legend and Title
# ---------------------------

# Create a shared legend
legend_plot <- ggplot(df_filtered, aes(x = Distance, y = Regret, color = Policy, fill = Policy)) +
  geom_point(alpha = 0.3) +
  geom_ribbon(data = quantile_bins_all, 
              aes(x = Distance, ymin = ci_lower, ymax = ci_upper, fill = Policy), 
              alpha = 0.3,
              inherit.aes = FALSE) +
  guides(color = guide_legend(title = "Policy"), 
         fill = guide_legend(title = "Policy")) +
  theme_minimal() +
  theme(legend.position = "right")

# Extract the legend
shared_legend <- get_legend(legend_plot)

# Arrange plots in a 2x2 grid
combined_plot <- (plot_list[[1]] | plot_list[[2]]) / 
  (plot_list[[3]] | plot_list[[4]]) +
  plot_layout(guides = "collect") & theme(legend.position = "right")

# Add the shared legend and global title
combined_plot <- combined_plot + plot_annotation(
  # title = "Regret vs. Distance with Log-transformed Quantile Regression 50% Confidence Bands",
  theme = theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold")),
  tag_levels = list(paste0("#training envs: ", c("2","4", "6", "8")))
) & theme(legend.position = "right")

# Display the combined plot
# print(combined_plot)

# ---------------------------
# 8. Save the Combined Plot (Optional)
# ---------------------------

# Save the combined plot to a PNG file
ggsave("regret_distance_quantile_regression_confidence_bands_all_nenv.png",
       plot = combined_plot,
       width = 12, height = 7.5, dpi = 300)

# -------------------------------------------------
# End of Corrected Script: Quantile Regression for Multiple n_envs
# -------------------------------------------------
