library(readxl)
library(rstan)
library(loo)
library(ggplot2)

# ---- データ読み込み・整形 ----
df <- read_excel("Rater6_v2.xlsx")
df$StudentID <- as.integer(as.factor(df$StudentID))
df$ItemID    <- as.integer(as.factor(df$ItemID))
df$TimeID    <- as.integer(as.factor(df$TimeID))
min_score    <- min(df$Score, na.rm=TRUE)
df$ScoreStan <- df$Score - min_score + 1

J <- max(df$StudentID)
I <- max(df$ItemID)
T <- max(df$TimeID)
K <- max(df$ScoreStan)
N <- nrow(df)

stan_data <- list(
  N  = N,
  J  = J,
  I  = I,
  T  = T,
  K  = K,
  jj = df$StudentID,
  ii = df$ItemID,
  tt = df$TimeID,
  y  = df$ScoreStan
)

# ---- Stan実行 ----
fit <- stan(
  file = "usami2010_gpcm1rater.stan",
  data = stan_data,
  seed = 123,
  iter = 2000,
  warmup = 1000,
  chains = 4,
  control = list(adapt_delta = 0.95)
)

# ---- パラメータ抽出・可視化 ----
beta_mean <- apply(extract(fit, pars = "beta")$beta, 2, mean)
beta_df   <- data.frame(TimeID = 1:length(beta_mean), Beta_Estimate = beta_mean)
ggplot(beta_df, aes(x = TimeID, y = Beta_Estimate)) +
  geom_line() + geom_point() +
  labs(title = "Rater Severity Drift (Beta) over Time",
       x = "Time Point", y = "Beta_Estimate") +
  theme_minimal()

# ---- モデル適合指標 ----
log_lik <- extract(fit, pars = "log_lik")$log_lik
loo_result  <- loo(log_lik)
waic_result <- waic(log_lik)
print(loo_result)
print(waic_result)