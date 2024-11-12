# -------------------------------------------------
# Comprehensive Script: Quantile Smoothing Splines (rqss)
# for Multiple n_env Values with 2x2 Grid Visualization
# -------------------------------------------------

# By Deng, Ruizhe and ChatGPT

# ---------------------------
# 1. Install and Load Packages
# ---------------------------

# Define required packages
required_packages <- c("ggplot2", "dplyr", "quantreg", "tidyr", "readr")

# Install any missing packages
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load the packages
library(ggplot2)
library(dplyr)
library(quantreg)
library(tidyr)
library(readr)

# ---------------------------
# 2. Read and Prepare Data
# ---------------------------

# Read the CSV file into a data frame
# Ensure 'regret_output.csv' is in your working directory
# If it's located elsewhere, provide the full path, e.g., "path/to/regret_output.csv"
df <- read_csv("regret_output.csv")

# Inspect the first few rows
cat("First few rows of the dataset:\n")
print(head(df))

# Convert columns to appropriate data types
df <- df %>%
  mutate(
    Distance = as.numeric(Distance),
    Regret = as.numeric(Regret),
    Policy = as.factor(Policy),
    n_env = as.factor(n_env)
  ) #%>% head(12000)

# Verify the structure of the data
cat("\nStructure of the dataset:\n")
print(str(df))

# ---------------------------
# 3. Filter Data for Relevant n_env Values
# ---------------------------

# Define the n_env values you want to analyze
# Assuming there are 4 unique n_env values for a 2x2 grid
# Modify as needed based on your data
desired_nenv <- c("2", "4", "6", "8")  # Example: "1", "2", "3", "4"

# Filter data for the desired n_env values
df_filtered <- df %>%
  filter(n_env %in% desired_nenv)

# Verify the number of observations
cat("\nNumber of observations after filtering for n_env =", 
    paste(desired_nenv, collapse = ", "), ":", nrow(df_filtered), "\n")

# ---------------------------
# 4. Check and Handle Missing Values
# ---------------------------

# Check for NA values in critical columns
na_counts <- df_filtered %>%
  summarise(
    Distance_NA = sum(is.na(Distance)),
    Regret_NA = sum(is.na(Regret)),
    Policy_NA = sum(is.na(Policy)),
    n_env_NA = sum(is.na(n_env))
  )

cat("\nNA counts in critical columns:\n")
print(na_counts)

# Remove rows with NA in Distance, Regret, Policy, or n_env
df_filtered <- df_filtered %>%
  filter(!is.na(Distance) & !is.na(Regret) & !is.na(Policy) & !is.na(n_env))

# Verify again
cat("\nAfter removing NAs, number of observations:", nrow(df_filtered), "\n")

# ---------------------------
# 5. Define a Grid of Distance Values
# ---------------------------

# Define a grid of Distance values for prediction
# The grid is defined per n_env to accommodate different ranges
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
# 6. Fit rqss Models for Each n_env and Policy
# ---------------------------

# Get the list of unique n_env and Policies
nenvs <- unique(df_filtered$n_env)
policies <- unique(df_filtered$Policy)

# Initialize an empty list to store predictions
quantile_predictions_all <- list()

# Loop over each n_env
for(env in nenvs){
  
  cat("\nProcessing n_env =", env, "\n")
  
  # Subset data for the current n_env
  df_env <- df_filtered %>%
    filter(n_env == env)
  
  # Get unique Policies within this n_env
  policies_env <- unique(df_env$Policy)
  
  # Loop over each Policy within this n_env
  for(p in policies_env){
    
    cat("  Fitting rqss models for Policy:", p, "\n")
    
    # Subset data for the current Policy
    data_p <- df_env %>%
      filter(Policy == p)
    
    # Define the grid for this n_env
    distance_grid_env <- distance_grids[[as.character(env)]]
    
    # Fit rqss model for tau = 0.025 (lower quantile)
    model_lower <- tryCatch({
      rqss(Regret ~ qss(Distance), tau = 0.25, data = data_p)
    }, error = function(e){
      message(paste("    Error in fitting lower quantile rqss model for Policy:", p, "in n_env:", env, "-", e$message))
      return(NULL)
    })
    
    # Fit rqss model for tau = 0.975 (upper quantile)
    model_upper <- tryCatch({
      rqss(Regret ~ qss(Distance), tau = 0.75, data = data_p)
    }, error = function(e){
      message(paste("    Error in fitting upper quantile rqss model for Policy:", p, "in n_env:", env, "-", e$message))
      return(NULL)
    })
    
    # Proceed only if both models are successfully fitted
    if(!is.null(model_lower) & !is.null(model_upper)){
      
      # Create a new data frame for prediction
      new_data <- data.frame(Distance = distance_grid_env)
      
      # Predict lower quantile
      preds_lower <- predict(model_lower, newdata = new_data)
      
      # Predict upper quantile
      preds_upper <- predict(model_upper, newdata = new_data)
      
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
      message(paste("    Skipping Policy:", p, "in n_env:", env, "due to model fitting errors."))
    }
  }
}

# Combine all predictions into a single data frame
quantile_bins_all <- bind_rows(quantile_predictions_all)

# Inspect the first few rows
cat("\nFirst few rows of the combined quantile predictions:\n")
print(head(quantile_bins_all))

# ---------------------------
# 7. Visualize the Data with Confidence Bands
# ---------------------------

# Plot with rqss confidence bands using facet_wrap for a 2x2 grid
plot_rqss_all <- ggplot() +
  # Scatter plot of the actual data
  geom_point(data = df_filtered, 
             aes(x = Distance, y = Regret, color = Policy), 
             alpha = 0.05) +
  
  # Shaded confidence bands
  geom_ribbon(data = quantile_bins_all, 
              aes(x = Distance, ymin = ci_lower, ymax = ci_upper, fill = Policy), 
              alpha = 0.2) +
  
  # Quantile regression lines (optional: add lines for lower and upper quantiles)
  geom_line(data = quantile_bins_all, 
            aes(x = Distance, y = ci_lower, color = Policy), 
            linetype = "dashed") +
  geom_line(data = quantile_bins_all, 
            aes(x = Distance, y = ci_upper, color = Policy), 
            linetype = "dashed") +
  
  # Facet by n_env to create a 2x2 grid (adjust nrow and ncol as needed)
  facet_wrap(~ n_env, nrow = 2, ncol = 2, scales = "free_x") +
  
  # Labels and titles
  labs(
    title = "Regret vs. Distance with rqss 50% Confidence Bands by Policy and n_env",
    x = "Distance",
    y = "Regret",
    color = "Policy",
    fill = "Policy"
  ) +
  xlim(5, 40) +
  # Theme for aesthetics
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
    axis.title = element_text(size = 14),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    strip.text = element_text(size = 14, face = "bold")  # Facet labels
  )

# Display the plot
print(plot_rqss_all)

# ---------------------------
# 8. Save the Plot (Optional)
# ---------------------------

# Save the plot to a PNG file
ggsave("regret_distance_rqss_confidence_bands_all_nenv.jpg",
       plot = plot_rqss_all,
       width = 16, height = 12, dpi = 300)

# -------------------------------------------------
# End of Comprehensive Script: rqss for Multiple n_envs
# -------------------------------------------------
