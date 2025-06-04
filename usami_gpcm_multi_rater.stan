data {
  int<lower=1> N;        // 観測数
  int<lower=1> I;        // 受験者数
  int<lower=1> J;        // 項目数
  int<lower=1> R;        // 評価者数
  int<lower=1> T;        // 時点数
  int<lower=1> K;        // カテゴリ数
  int<lower=1,upper=I> ii[N];
  int<lower=1,upper=J> jj[N];
  int<lower=1,upper=R> rr[N];
  int<lower=1,upper=T> tt[N];  // 各観測のTimeID（時点）
  int<lower=1,upper=K> y[N];
}
parameters {
  vector[I] theta;              
  matrix[J, R] alpha_log;       
  matrix[J, R] delta;           
  array[J, R, K-1] real tau_raw; // K-1次元のみ推定、1つ目は0固定
  matrix[R, T] beta;             // Rater×TimeIDの厳しさドリフト
  real<lower=0> sigma;           // driftランダムウォーク変動幅
}
transformed parameters {
  array[J, R, K] real tau;
  for (j in 1:J)
    for (r in 1:R) {
      tau[j, r, 1] = 0;
      for (k in 2:K)
        tau[j, r, k] = tau_raw[j, r, k-1];
    }
  matrix[J, R] alpha = exp(alpha_log);
}
model {
  // 事前分布
  theta ~ normal(0, 1);
  to_vector(alpha_log) ~ normal(0, 0.5);
  to_vector(delta) ~ normal(0, 2);
  for (j in 1:J)
    for (r in 1:R)
      for (k in 1:(K-1))
        tau_raw[j, r, k] ~ normal(0, 2);

  // drift（ランダムウォーク）
  for (r in 1:R) {
    beta[r, 1] ~ normal(0, 1);
    for (t in 2:T)
      beta[r, t] ~ normal(beta[r, t-1], sigma);
  }
  sigma ~ lognormal(-3, 1);

  // 尤度
  for (n in 1:N) {
    int j = jj[n];
    int r = rr[n];
    int i = ii[n];
    int t = tt[n];
    vector[K] num;
    for (k in 1:K) {
      real s = alpha[j, r] * (
        (k-1)*(theta[i] - delta[j, r] - beta[r, t]) - sum(tau[j, r, 1:k])
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
    int r = rr[n];
    int i = ii[n];
    int t = tt[n];
    vector[K] num;
    for (k in 1:K) {
      real s = alpha[j, r] * (
        (k-1)*(theta[i] - delta[j, r] - beta[r, t]) - sum(tau[j, r, 1:k])
      );
      num[k] = s;
    }
    log_lik[n] = num[y[n]] - log_sum_exp(num);
  }
}
