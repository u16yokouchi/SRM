library(readxl)
library(rstan)
library(loo)
library(ggplot2)

# ---- データ前処理 ----
df <- read_excel("Rater6_v2.xlsx")
df$StudentID <- as.integer(as.factor(df$StudentID))
df$TimeID <- as.integer(as.factor(df$TimeID))
df$ItemID <- as.integer(as.factor(df$ItemID))   # 項目列名が違えば適宜修正
min_score <- min(df$Score, na.rm=TRUE)
df$ScoreStan <- df$Score - min_score + 1

# θ=0固定対象
unique_time <- sort(unique(df$TimeID))
fixed_j <- sapply(unique_time, function(t) subset(df, TimeID == t)$StudentID[1])
is_fixed <- as.integer(1:max(df$StudentID) %in% fixed_j)

# Stan用データリスト
J <- max(df$StudentID)
I <- max(df$ItemID)
T <- max(df$TimeID)
K <- max(df$ScoreStan)
N <- nrow(df)
n_fixed <- length(fixed_j)
J_free <- J - n_fixed
stan_data <- list(
  N = N, J = J, I = I, T = T, K = K,
  jj = df$StudentID,
  ii = df$ItemID,
  tt = df$TimeID,
  y  = df$ScoreStan,
  n_fixed = n_fixed,
  fixed_j = fixed_j,
  J_free = J_free,
  is_fixed = is_fixed
)

# ---- Stanモデル推定 ----
fit <- stan(
  file = "v2_mfrm_single_rater.stan", 
  data = stan_data,
  seed = 123,
  iter = 2000,
  warmup = 1000,
  chains = 4,
  control = list(adapt_delta = 0.95)
)

# ---- パラメータ抽出 ----
beta_mean <- apply(extract(fit, pars = "beta")$beta, 2, mean)
beta_df <- data.frame(TimeID = 1:length(beta_mean), Beta_Estimate = beta_mean)

theta_mean <- apply(extract(fit, pars = "theta")$theta, 2, mean)
theta_summary <- summary(fit, pars = "theta")$summary
theta_df <- data.frame(
  StudentID = 1:length(theta_mean),
  Theta_Estimate = theta_mean,
  Rhat = theta_summary[, "Rhat"],
  n_eff = theta_summary[, "n_eff"]
)

b_mean <- apply(extract(fit, pars = "b")$b, 2, mean)
b_summary <- summary(fit, pars = "b")$summary
b_df <- data.frame(
  ItemID = 1:length(b_mean),
  b_Estimate = b_mean,
  Rhat = b_summary[, "Rhat"],
  n_eff = b_summary[, "n_eff"]
)

# ---- βのグラフ化 ----
ggplot(beta_df, aes(x = TimeID, y = Beta_Estimate)) +
  geom_line(size=1) +
  geom_point(size=2) +
  labs(title = "Rater Severity Drift (Beta) over Time",
       x = "Time Point",
       y = "Beta_Estimate") +
  theme_minimal()

# ---- WAIC・LOO ----
log_lik <- extract(fit, pars = "log_lik")$log_lik
loo_result <- loo(log_lik)
waic_result <- waic(log_lik)
print(loo_result)
print(waic_result)

# ---- Posterior Predictive p-value（カテゴリ分布での乖離を指標に） ----
obs_cat <- table(df$ScoreStan) / nrow(df)  # 観測カテゴリ分布

# 関数：各応答ごとカテゴリ確率のモデル予測値（b[ii[n]]込み）
calc_pred_prob <- function(theta, b, beta, d, jj, ii, tt, K) {
  N <- length(jj)
  pred_prob <- matrix(0, N, K)
  for (n in 1:N) {
    cprob <- numeric(K)
    for (k in 1:K) {
      s <- 0
      for (m in 1:k) {
        s <- s + theta[jj[n]] - b[ii[n]] - beta[tt[n]] - d[m]
      }
      cprob[k] <- s
    }
    pred_prob[n, ] <- exp(cprob - log(sum(exp(cprob))))
  }
  colMeans(pred_prob)
}

# Stan推定値の平均で予測分布計算
d_mean <- apply(extract(fit, pars = "d")$d, 2, mean)
b_mean <- apply(extract(fit, pars = "b")$b, 2, mean)
theta_mean <- apply(extract(fit, pars = "theta")$theta, 2, mean)
beta_mean <- apply(extract(fit, pars = "beta")$beta, 2, mean)

pred_cat <- calc_pred_prob(theta_mean, b_mean, beta_mean, d_mean,
                           df$StudentID, df$ItemID, df$TimeID, K)

ppp <- sum(abs(pred_cat - as.numeric(obs_cat)))
cat("Posterior predictive p-value (差の合計):", ppp, "\n")

# ---- RMSE・Intra-rater reliability（Accuracy） ----
# 関数：各応答ごと予測カテゴリ（b[ii[n]]込み）
pred_cat_each <- function(theta, b, beta, d, jj, ii, tt, K) {
  N <- length(jj)
  y_pred <- integer(N)
  for (n in 1:N) {
    cprob <- numeric(K)
    for (k in 1:K) {
      s <- 0
      for (m in 1:k) {
        s <- s + theta[jj[n]] - b[ii[n]] - beta[tt[n]] - d[m]
      }
      cprob[k] <- s
    }
    prob <- exp(cprob - log(sum(exp(cprob))))
    y_pred[n] <- which.max(prob)
  }
  y_pred
}

y_pred <- pred_cat_each(theta_mean, b_mean, beta_mean, d_mean,
                        df$StudentID, df$ItemID, df$TimeID, K)
RMSE <- sqrt(mean((y_pred - df$ScoreStan)^2))
cat("RMSE:", RMSE, "\n")

accuracy <- mean(y_pred == df$ScoreStan)
cat("Intra-rater reliability (Accuracy):", accuracy, "\n")