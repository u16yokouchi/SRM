data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> I;
  int<lower=1> T;
  int<lower=1> K;
  int<lower=1,upper=J> jj[N];
  int<lower=1,upper=I> ii[N];
  int<lower=1,upper=T> tt[N];
  int<lower=1,upper=K> y[N];
}
parameters {
  vector[J] alpha_log;             // 項目ごと識別力（logで推定、expで戻す）
  vector[J] delta;                 // 項目困難度
  array[J, K-1] real tau_raw;      // カテゴリ閾値（K-1次元のみ推定、1つは0固定）
  vector[I] theta;                 // 受験者能力
  vector[T] beta;                  // 時点ごとのdrift（評価の厳しさ）
  real<lower=0> sigma;             // driftランダムウォーク変動幅
}
transformed parameters {
  array[J, K] real tau;
  for (j in 1:J) {
    tau[j, 1] = 0;
    for (k in 2:K)
      tau[j, k] = tau_raw[j, k-1];
  }
  vector[J] alpha = exp(alpha_log);
}
model {
  theta ~ normal(0, 1);
  alpha_log ~ normal(0, 0.5);
  delta ~ normal(0, 2);
  for (j in 1:J)
    for (k in 1:(K-1))
      tau_raw[j, k] ~ normal(0, 2);
  beta[1] ~ normal(0, 1);
  for (t in 2:T)
    beta[t] ~ normal(beta[t-1], sigma);
  sigma ~ lognormal(-3, 1);

  for (n in 1:N) {
    int j = jj[n];
    int i = ii[n];
    int t = tt[n];
    vector[K] num;
    for (k in 1:K) {
      real s = alpha[j] * (
        (k-1)*(theta[i] - delta[j] - beta[t]) - sum(tau[j, 1:k])
      );
      num[k] = s;
    }
    target += num[y[n]] - log_sum_exp(num);
  }
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    int j = jj[n];
    int i = ii[n];
    int t = tt[n];
    vector[K] num;
    for (k in 1:K) {
      real s = exp(alpha_log[j]) * (
        (k-1)*(theta[i] - delta[j] - beta[t]) - sum(tau[j, 1:k])
      );
      num[k] = s;
    }
    log_lik[n] = num[y[n]] - log_sum_exp(num);
  }
}
