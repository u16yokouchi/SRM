library(readxl)
library(rstan)
library(loo)
library(ggplot2)
library(reshape2)

# 1. データ読み込み・前処理
df <- read_excel("20250604_RSD_Full.xlsx")
df$StudentID <- as.integer(as.factor(df$StudentID))
df$ItemID    <- as.integer(as.factor(df$ItemID))
df$RaterID   <- as.integer(as.factor(df$RaterID))
df$TimeID    <- as.integer(as.factor(df$TimeID))
df$ScoreStan <- df$Score + 1

N <- nrow(df)
I <- max(df$StudentID)
J <- max(df$ItemID)
R <- max(df$RaterID)
T <- max(df$TimeID)
K <- max(df$ScoreStan)

stan_data <- list(
  N = N, I = I, J = J, R = R, T = T, K = K,
  ii = df$StudentID,
  jj = df$ItemID,
  rr = df$RaterID,
  tt = df$TimeID,
  y  = df$ScoreStan
)

# 2. Stan実行（usami_gpcm_multi_rater.stanを保存しておくこと）
fit <- stan(
  file = "usami_gpcm_multi_rater.stan",
  data = stan_data,
  iter = 2000, warmup = 1000, chains = 4, seed = 123,
  control = list(adapt_delta = 0.95)
)

# 3. Rater×TimeID drift（beta）の事後平均行列
beta_post <- extract(fit, pars="beta")$beta  # iterations × R × T
beta_mean <- apply(beta_post, c(2,3), mean)  # R × T行列
colnames(beta_mean) <- paste0("Time", 1:T)
rownames(beta_mean) <- paste0("Rater", 1:R)

# 4. 全Rater分まとめて描画（横軸：TimeID、縦軸：Beta_Estimate）
beta_long <- melt(beta_mean)
colnames(beta_long) <- c("RaterID", "TimeID", "Beta_Estimate")
ggplot(beta_long, aes(x = as.numeric(gsub("Time", "", TimeID)), y = Beta_Estimate, color=RaterID, group=RaterID)) +
  geom_line() + geom_point() +
  labs(title="Rater-wise Severity Drift (Beta) over Time",
       x="TimeID", y="Rater Severity (Beta)") +
  theme_minimal()

# 5. Rater6のみ抜き出し、個別プロット
rater6_id <- 6
plot(beta_mean[rater6_id, ], type="b", main="Rater6 Drift (Beta) over Time", xlab="TimeID", ylab="Beta")

# 6. モデル適合度
log_lik <- extract(fit, pars = "log_lik")$log_lik
loo_result  <- loo(log_lik)
waic_result <- waic(log_lik)
print(loo_result)
print(waic_result)