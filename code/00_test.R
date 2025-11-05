project_name <- "Bank Marketing Campaign Prediction"
team_members <- c("Mattia", "Eliya", "Adrian")
date_started <- Sys.Date()

cat("=== PROJECT INITIALIZED ===\n")
cat("Project:", project_name, "\n")
cat("Team:", paste(team_members, collapse = ", "), "\n")
cat("Date:", as.character(date_started), "\n")
cat("Git integration: WORKING ✓\n")

test_data <- data.frame(
  id = 1:5,
  value = rnorm(5)
)

print(test_data)
