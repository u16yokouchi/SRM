library(readxl)
library(rstan)

# データ読み込み
df <- read_excel("Rater6.xlsx")
df$StudentID <- as.integer(as.factor(df$StudentID))
df$TimeID <- as.integer(as.factor(df$TimeID))

# Score（カテゴリ）はStanで1始まり必須
min_score <- min(df$Score, na.rm=TRUE)
df$ScoreStan <- df$Score - min_score + 1

# 各TimeIDで初めて登場するStudentIDを特定
unique_time <- sort(unique(df$TimeID))
fixed_j <- sapply(unique_time, function(t) subset(df, TimeID == t)$StudentID[1])
is_fixed <- as.integer(1:max(df$StudentID) %in% fixed_j)

J <- max(df$StudentID)
T <- max(df$TimeID)
K <- max(df$ScoreStan)
N <- nrow(df)
n_fixed <- length(fixed_j)
J_free <- J - n_fixed

stan_data <- list(
  N = N,
  J = J,
  T = T,
  K = K,
  jj = df$StudentID,
  tt = df$TimeID,
  y  = df$ScoreStan,
  n_fixed = n_fixed,
  fixed_j = fixed_j,
  J_free = J_free,
  is_fixed = is_fixed
)

fit <- stan(
  file = "mfrm_single_rater.stan",
  data = stan_data,
  seed = 123,
  iter = 2000,
  warmup = 1000,
  chains = 4,
  control = list(adapt_delta = 0.95)
)

# betaのタイムポイントごとの推定値（事後平均）を出力
beta_posterior <- extract(fit, pars = "beta")$beta
beta_mean <- apply(beta_posterior, 2, mean)
beta_df <- data.frame(TimeID = 1:length(beta_mean), Beta_Estimate = beta_mean)
print(beta_df)

# thetaの事後平均・Rhat・n_eff
theta_posterior <- extract(fit, pars = "theta")$theta
theta_mean <- apply(theta_posterior, 2, mean)
theta_summary <- summary(fit, pars = "theta")$summary
theta_df <- data.frame(
  StudentID = 1:ncol(theta_posterior),
  Theta_Estimate = theta_mean,
  Rhat = theta_summary[, "Rhat"],
  n_eff = theta_summary[, "n_eff"]
)
print(theta_df)

# betaのタイムポイントごとの事後平均（既出）
beta_posterior <- extract(fit, pars = "beta")$beta
beta_mean <- apply(beta_posterior, 2, mean)
beta_df <- data.frame(TimeID = 1:length(beta_mean), Beta_Estimate = beta_mean)
print(beta_df)

# loo/waic計算（Stanでgenerated quantities{vector[N] log_lik;}が必要）
library(loo)
log_lik <- extract(fit, pars = "log_lik")$log_lik
loo_result <- loo(log_lik)
waic_result <- waic(log_lik)
print(loo_result)
print(waic_result)