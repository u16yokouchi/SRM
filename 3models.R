library(readxl)
library(rstan)
library(loo)
library(ggplot2)
library(reshape2)

# ========== 1. データ読み込み ==========

# --- GRM/GPCM: Rater6_v2.xlsx（Rater6のみデータ） ---
df_rater6 <- read_excel("Rater6_v2.xlsx")
df_rater6$StudentID <- as.integer(as.factor(df_rater6$StudentID))
df_rater6$ItemID    <- as.integer(as.factor(df_rater6$ItemID))
df_rater6$TimeID    <- as.integer(as.factor(df_rater6$TimeID))
df_rater6$ScoreStan <- df_rater6$Score + 1
N <- nrow(df_rater6)
J <- max(df_rater6$StudentID)
I <- max(df_rater6$ItemID)
T <- max(df_rater6$TimeID)
K <- max(df_rater6$ScoreStan)
stan_data_rater6 <- list(
  N = N, J = J, I = I, T = T, K = K,
  jj = df_rater6$StudentID,
  ii = df_rater6$ItemID,
  tt = df_rater6$TimeID,
  y  = df_rater6$ScoreStan
)

# --- Usami(2010): 20250604_RSD_Full.xlsx からRater6のみ抽出 ---
df_full <- read_excel("20250604_RSD_Full.xlsx")
df_full$StudentID <- as.integer(as.factor(df_full$StudentID))
df_full$ItemID    <- as.integer(as.factor(df_full$ItemID))
df_full$RaterID   <- as.integer(as.factor(df_full$RaterID))
df_full$TimeID    <- as.integer(as.factor(df_full$TimeID))
df_full$ScoreStan <- df_full$Score + 1

df_r6 <- subset(df_full, RaterID == 6)
N6 <- nrow(df_r6)
I6 <- max(df_r6$StudentID)
J6 <- max(df_r6$ItemID)
T6 <- max(df_r6$TimeID)
K6 <- max(df_r6$ScoreStan)
stan_data_usami <- list(
  N = N6, I = I6, J = J6, R = 1, T = T6, K = K6,
  ii = df_r6$StudentID,
  jj = df_r6$ItemID,
  rr = rep(1, N6),
  tt = df_r6$TimeID,
  y  = df_r6$ScoreStan
)

# ========== 2. Stan推定 ==========

fit_grm   <- stan(file = "grm_drift_single_rater.stan",   data = stan_data_rater6, iter = 2000, warmup = 1000, chains = 4, seed = 1)
fit_gpcm  <- stan(file = "gpcm_drift_single_rater.stan",  data = stan_data_rater6, iter = 2000, warmup = 1000, chains = 4, seed = 1)
fit_usami <- stan(file = "usami_gpcm_multi_rater.stan",   data = stan_data_usami,  iter = 2000, warmup = 1000, chains = 4, seed = 1)

# ========== 3. 各モデルのβt/waic/loo/pppを抽出 ==========

extract_beta <- function(fit, param = "beta") {
  if (!(param %in% names(rstan::extract(fit)))) return(NULL)
  beta_post <- extract(fit, pars = param)[[param]]
  if (is.null(beta_post)) return(NULL)
  if (is.matrix(beta_post)) { apply(beta_post, 2, mean) }
  else if (length(dim(beta_post)) == 3) { apply(beta_post, 3, mean) } # R × T型ならR=1
  else as.numeric(beta_post)
}

beta_grm   <- extract_beta(fit_grm,   "beta")
beta_gpcm  <- extract_beta(fit_gpcm,  "beta")
beta_usami <- extract_beta(fit_usami, "beta")

# どれか一つでもNULL/長さゼロならストップ
if (is.null(beta_grm) | is.null(beta_gpcm) | is.null(beta_usami) |
    length(beta_grm)==0 | length(beta_gpcm)==0 | length(beta_usami)==0) {
  stop("いずれかのモデルのbetaが抽出できていません（Stan出力・extract箇所を要確認）")
}

T_grm   <- length(beta_grm)
T_gpcm  <- length(beta_gpcm)
T_usami <- length(beta_usami)
beta_df <- rbind(
  data.frame(TimeID = 1:T_grm,   Beta = beta_grm,   Model = "GRM"),
  data.frame(TimeID = 1:T_gpcm,  Beta = beta_gpcm,  Model = "GPCM"),
  data.frame(TimeID = 1:T_usami, Beta = beta_usami, Model = "Usami2010")
)

# ========== 4. WAIC/LOO/PPP計算（pppは適宜関数を拡張） ==========

get_waic_loo_ppp <- function(fit, df, theta_mean=NULL, beta_mean=NULL, d_mean=NULL, ...) {
  log_lik <- extract(fit, pars = "log_lik")$log_lik
  waic_val <- waic(log_lik)
  loo_val  <- loo(log_lik)
  obs_cat <- table(df$ScoreStan) / nrow(df)
  list(waic = waic_val, loo = loo_val, ppp = NA)
}

res_grm   <- get_waic_loo_ppp(fit_grm,   df_rater6)
res_gpcm  <- get_waic_loo_ppp(fit_gpcm,  df_rater6)
res_usami <- get_waic_loo_ppp(fit_usami, df_r6)

# ========== 5. プロット＆結果表 ==========

# βtの比較プロット
ggplot(beta_df, aes(x = TimeID, y = Beta, color = Model, group = Model)) +
  geom_line(size=1) +
  geom_point(size=2) +
  geom_text(aes(label=Model), vjust=-1, size=3, show.legend=FALSE, check_overlap = TRUE) +
  labs(title = "Rater6 厳しさ drift (βt) 各モデル比較",
       x = "TimeID", y = "Beta (drift厳しさ)", color="Model") +
  theme_minimal()

# 結果のテーブル出力例
waic_vec <- c(res_grm$waic$estimates["waic", "Estimate"],
              res_gpcm$waic$estimates["waic", "Estimate"],
              res_usami$waic$estimates["waic", "Estimate"])
loo_vec  <- c(res_grm$loo$estimates["looic", "Estimate"],
              res_gpcm$loo$estimates["looic", "Estimate"],
              res_usami$loo$estimates["looic", "Estimate"])
ppp_vec  <- c(res_grm$ppp, res_gpcm$ppp, res_usami$ppp)

results_df <- data.frame(
  Model = c("GRM", "GPCM", "Usami2010"),
  WAIC  = waic_vec,
  LOOIC = loo_vec,
  PPP   = ppp_vec
)
print(results_df)