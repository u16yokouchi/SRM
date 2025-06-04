data {
  int<lower=1> N;         // 観測数
  int<lower=1> J;         // 受験者数
  int<lower=1> I;         // 項目数
  int<lower=1> T;         // 時点数
  int<lower=1> K;         // カテゴリ数
  int<lower=1,upper=J> jj[N]; // 各応答の受験者
  int<lower=1,upper=I> ii[N]; // 各応答の項目
  int<lower=1,upper=T> tt[N]; // 各応答の時点
  int<lower=1,upper=K> y[N];  // 各応答のカテゴリ（1始まり）
}
parameters {
  vector[J] theta;              // 受験者能力
  vector[I] delta;              // 項目困難度
  vector<lower=0>[I] alpha;     // 項目識別力
  matrix[I, K] tau_raw;         // 閾値パラメータ
  vector[T] beta;               // 時点drift
  real<lower=0> sigma;          // drift変動幅
}
transformed parameters {
  matrix[I, K] tau;
  // 各項目jごとにsum_m tau_jm = 0制約
  for (i in 1:I) {
    real tau_mean = mean(tau_raw[i,]);
    for (k in 1:K) tau[i, k] = tau_raw[i, k] - tau_mean;
  }
}
model {
  // 事前分布
  theta ~ normal(0, 1);
  delta ~ normal(0, 1);
  alpha ~ lognormal(0, 0.5);
  to_vector(tau_raw) ~ normal(0, 1);
  beta[1] ~ normal(0, 1);
  for (t in 2:T) beta[t] ~ normal(beta[t-1], sigma);
  sigma ~ lognormal(-3, 1);

  // 尤度
  for (n in 1:N) {
    vector[K] numerators;
    for (k in 1:K) {
      real summ = 0;
      for (m in 1:k)
        summ += tau[ii[n], m];
      numerators[k] = alpha[ii[n]] * ((k-1)*(theta[jj[n]] - delta[ii[n]] - beta[tt[n]]) - summ);
    }
    target += numerators[y[n]] - log_sum_exp(numerators);
  }
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    vector[K] numerators;
    for (k in 1:K) {
      real summ = 0;
      for (m in 1:k)
        summ += tau[ii[n], m];
      numerators[k] = alpha[ii[n]] * ((k-1)*(theta[jj[n]] - delta[ii[n]] - beta[tt[n]]) - summ);
    }
    log_lik[n] = numerators[y[n]] - log_sum_exp(numerators);
  }
}
