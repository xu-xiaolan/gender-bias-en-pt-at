# Mixed Effects Logistic Regression
# Gender bias in MT: English to European Portuguese
# Requires: lme4, dplyr, broom.mixed
# install.packages(c("lme4", "dplyr", "broom.mixed"))

library(lme4)
library(dplyr)
library(broom.mixed)

# ── 1. Load data (already in long format) ─────────────────────────────────────
long_df <- read.csv("slate_long_clean.csv", stringsAsFactors = FALSE)

# ── 2. Encode predictors ───────────────────────────────────────────────────────
# Reference levels: NMT, male (gender_F=0)
long_df$source_gender <- relevel(factor(long_df$source_gender), ref = "male")
long_df$system_type   <- relevel(factor(long_df$system_type),   ref = "NMT")

cat("Data shape:", nrow(long_df), "rows,", ncol(long_df), "cols\n")
cat("Correct translations:", sum(long_df$correct), "/", nrow(long_df), "\n\n")

# ── 3. Fit mixed effects logistic regression ───────────────────────────────────
# Fixed effects: source gender, system type
# Random effect: sentence (controls for sentence-level difficulty)
model <- glmer(
  correct ~ gender_F + system_type + (1 | sentence_id),
  data    = long_df,
  family  = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

# ── 4. Results table with OR and 95% CI ───────────────────────────────────────
cat("=== Mixed Effects Logistic Regression Results ===\n")
cat("Reference: NMT, male gender\n\n")

tidy_res <- tidy(model, conf.int = TRUE, exponentiate = TRUE, effects = "fixed")
tidy_res <- tidy_res %>%
  filter(term != "(Intercept)") %>%
  mutate(
    stars = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01  ~ "**",
      p.value < 0.05  ~ "*",
      TRUE            ~ "n.s."
    ),
    label = recode(term,
                   "gender_F"        = "Target gender: F",
                   "system_typeLLM"  = "System type: LLM"
    )
  ) %>%
  select(label, estimate, conf.low, conf.high, p.value, stars)

colnames(tidy_res) <- c("Predictor", "OR", "CI_low", "CI_high", "p", "sig")

tidy_beta <- tidy(model, conf.int = TRUE, exponentiate = FALSE, effects = "fixed") %>%
  filter(term != "(Intercept)")

cat(sprintf("%-25s %6s %6s  %-14s %8s %5s\n",
            "Predictor", "β", "OR", "95% CI", "p", ""))
cat(strrep("-", 70), "\n")

for (i in 1:nrow(tidy_res)) {
  cat(sprintf("%-25s %6.3f %6.2f  [%.2f, %.2f]  %8.3f %5s\n",
              tidy_res$Predictor[i],
              tidy_beta$estimate[i],
              tidy_res$OR[i],
              tidy_res$CI_low[i],
              tidy_res$CI_high[i],
              tidy_res$p[i],
              tidy_res$sig[i]))
}

cat("\n")
cat("N =", nrow(long_df), "\n")
cat("AIC =", round(AIC(model), 1), "\n")
re_var <- as.data.frame(VarCorr(model))
cat("Random effect (sentence) variance =", round(re_var$vcov[1], 4), "\n")

