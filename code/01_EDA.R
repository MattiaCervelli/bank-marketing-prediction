library(tidyverse)
library(corrplot)
library(gridExtra)

if(!file.exists("data/bank-additional-full.csv")) {
  stop("ERROR: Data file not found!\n")
}

cat("=== LOADING DATA ===\n")
bank <- read.csv("data/bank-additional-full.csv", sep=";", stringsAsFactors = TRUE)

cat("=== DATASET OVERVIEW ===\n")
cat("Dimensions:", dim(bank)[1], "rows x", dim(bank)[2], "columns\n\n")

str(bank)

cat("\n=== FIRST 10 ROWS ===\n")
print(head(bank, 10))

cat("\n=== SUMMARY STATISTICS ===\n")
print(summary(bank))

cat("\n=== TARGET VARIABLE DISTRIBUTION ===\n")
target_counts <- table(bank$y)
target_props <- prop.table(target_counts) * 100

print(target_counts)
cat("\nProportions:\n")
print(round(target_props, 2))

imbalance_ratio <- target_props[1] / target_props[2]
cat("\nClass Imbalance Ratio:", round(imbalance_ratio, 2), ":1 (no:yes)\n")

target_plot <- ggplot(bank, aes(x = y, fill = y)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..), vjust=-0.5) +
  labs(title = "Target Variable Distribution (Subscription)",
       subtitle = paste0("Class Imbalance: ", 
                         round(target_props[2], 1), 
                         "% subscribed"),
       x = "Subscribed to Term Deposit", 
       y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("no" = "#E74C3C", "yes" = "#2ECC71")) +
  theme(legend.position = "none")

print(target_plot)

cat("\n⚠️  CLASS IMBALANCE DETECTED: Only", 
    round(target_props[2], 1), 
    "% of clients subscribed\n")
cat("📌 IMPLICATION: We'll need to address this in our methodology\n\n")

cat("\n=== MISSING VALUES (coded as 'unknown') ===\n")
missing_summary <- sapply(bank, function(x) {
  if(is.factor(x)) {
    sum(x == "unknown")
  } else {
    sum(is.na(x))
  }
})

missing_df <- data.frame(
  Variable = names(missing_summary),
  Missing_Count = missing_summary,
  Percentage = round(missing_summary / nrow(bank) * 100, 2)
)

missing_df <- missing_df[missing_df$Missing_Count > 0, ]

if(nrow(missing_df) > 0) {
  cat("Variables with 'unknown' values:\n")
  print(missing_df, row.names = FALSE)
} else {
  cat("No missing values detected!\n")
}

cat("\n=== NUMERICAL FEATURES SUMMARY ===\n")
numerical_vars <- c("age", "duration", "campaign", "pdays", "previous",
                    "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
                    "euribor3m", "nr.employed")

print(summary(bank[, numerical_vars]))

cat("\n IMPORTANT: 'duration' WILL BE EXCLUDED from predictive model\n")
cat("   Reason: Duration is only known AFTER the call (per dataset documentation)\n")
cat("   We'll use it only for benchmark comparison\n\n")

cat("=== CORRELATION ANALYSIS ===\n")
numerical_for_model <- numerical_vars[numerical_vars != "duration"]
cor_matrix <- cor(bank[, numerical_for_model])

corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, tl.cex = 0.8,
         title = "Correlation Matrix (Numerical Features - Duration Excluded)",
         mar = c(0,0,2,0))

cat("\n=== HIGH CORRELATIONS (|r| > 0.8) ===\n")
high_cor_indices <- which(abs(cor_matrix) > 0.8 & abs(cor_matrix) < 1, arr.ind = TRUE)

if(nrow(high_cor_indices) > 0) {
  high_cor_pairs <- data.frame(
    Var1 = rownames(cor_matrix)[high_cor_indices[,1]],
    Var2 = colnames(cor_matrix)[high_cor_indices[,2]],
    Correlation = cor_matrix[high_cor_indices]
  )
  # Remove duplicates (keep only one direction)
  high_cor_pairs <- high_cor_pairs[high_cor_pairs$Var1 < high_cor_pairs$Var2, ]
  
  if(nrow(high_cor_pairs) > 0) {
    print(high_cor_pairs, row.names = FALSE)
    cat("\IMPLICATION: May need to address multicollinearity (VIF analysis)\n")
  }
} else {
  cat("No high correlations detected (all |r| < 0.8)\n")
}

cat("\n\n=== CATEGORICAL FEATURES OVERVIEW ===\n")
categorical_vars <- c("job", "marital", "education", "default", 
                      "housing", "loan", "contact", "month", 
                      "day_of_week", "poutcome")

for(var in categorical_vars) {
  cat("\n", var, ":\n", sep="")
  print(table(bank[[var]]))
}

cat("DATA QUALITY:\n")
cat("   • Dataset size: 41,188 observations × 21 variables\n")
cat("   • Target: Binary (subscribed yes/no)\n")
cat("   • No missing values in numerical features\n")
cat("   • Some 'unknown' categories in categorical variables\n\n")

cat("PROBLEM CHARACTERISTICS:\n")
cat("   • Binary classification with class imbalance (~", round(target_props[2], 1), "% positive)\n", sep="")
cat("   • Real-world business problem: Marketing campaign optimization\n")
cat("   • Mix of client demographics, campaign data, and macro-economic indicators\n")
cat("   • Temporal dimension: Data spans May 2008 - Nov 2010\n\n")

cat("METHODOLOGICAL OPPORTUNITIES:\n")
cat("   • Class imbalance → Cost-sensitive learning / threshold optimization\n")
cat("   • Economic indicators → Interaction effects with client characteristics\n")
cat("   • Multiple categorical variables → Dummy encoding, interaction terms\n")
cat("   • Duration exclusion → Realistic predictive model (best practice)\n\n")

cat("ORIGINAL CONTRIBUTIONS WE CAN MAKE:\n")
cat("   1. Cost-sensitive threshold optimization for marketing ROI\n")
cat("   2. Interaction analysis: economic context × demographics\n")
cat("   3. Polynomial features for age, campaign, previous contacts\n")
cat("   4. Comparative k-fold CV analysis (k=5 vs k=10)\n")
cat("   5. Feature importance interpretation for business insights\n\n")

cat("=== CREATING KEY VISUALIZATIONS ===\n\n")

# Age distribution by target
p1 <- ggplot(bank, aes(x = age, fill = y)) +
  geom_histogram(bins = 30, position = "dodge", alpha = 0.7) +
  labs(title = "Age Distribution by Subscription",
       x = "Age", y = "Count") +
  theme_minimal() +
  scale_fill_manual(values = c("no" = "#E74C3C", "yes" = "#2ECC71"),
                    name = "Subscribed")

# Campaign contacts by target
p2 <- ggplot(bank, aes(x = y, y = campaign, fill = y)) +
  geom_boxplot() +
  labs(title = "Campaign Contacts by Subscription",
       x = "Subscribed", y = "Number of Contacts") +
  theme_minimal() +
  scale_fill_manual(values = c("no" = "#E74C3C", "yes" = "#2ECC71")) +
  theme(legend.position = "none")

# Job type distribution
p3 <- ggplot(bank, aes(x = reorder(job, job, function(x) -length(x)), fill = y)) +
  geom_bar(position = "fill") +
  coord_flip() +
  labs(title = "Subscription Rate by Job Type",
       x = "Job", y = "Proportion") +
  theme_minimal() +
  scale_fill_manual(values = c("no" = "#E74C3C", "yes" = "#2ECC71"),
                    name = "Subscribed")

# Economic indicator by target
p4 <- ggplot(bank, aes(x = y, y = euribor3m, fill = y)) +
  geom_violin() +
  labs(title = "Euribor Rate Distribution by Subscription",
       x = "Subscribed", y = "Euribor 3-month Rate (%)") +
  theme_minimal() +
  scale_fill_manual(values = c("no" = "#E74C3C", "yes" = "#2ECC71")) +
  theme(legend.position = "none")

# Display plots
grid.arrange(p1, p2, p3, p4, ncol = 2)