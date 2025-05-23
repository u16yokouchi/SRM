data {
  int<lower=1> N;               // 応答数
  int<lower=1> J;               // 受験者数
  int<lower=1> I;               // 項目数
  int<lower=1> J_free;          // θを推定する受験者数
  int<lower=1> T;               // 時点数
  int<lower=1> K;               // カテゴリ数

  int<lower=1,upper=J> jj[N];   // 応答ごとの受験者ID
  int<lower=1,upper=I> ii[N];   // 応答ごとの項目ID
  int<lower=1,upper=T> tt[N];   // 応答ごとの時点ID
  int<lower=1,upper=K> y[N];    // 応答ごとのスコア
  int<lower=1> n_fixed;
  int<lower=1,upper=J> fixed_j[n_fixed];
  int<lower=0,upper=1> is_fixed[J];
}
parameters {
  vector[J] theta;
  vector[I] b;
  vector[T] beta;
  vector[K] d;
  real<lower=0> sigma;
}
model {
  // θのうち0固定すべきものだけ強制的に0にする（それ以外は標準正規）
  for (j in 1:J) {
    if (is_fixed[j] == 1) {
      target += normal_lpdf(theta[j] | 0, 1e-8);
    } else {
      target += normal_lpdf(theta[j] | 0, 1);
    }
  }
  b ~ normal(0, 1);
  d ~ normal(0, 1);
  beta[1] ~ normal(0, 1);
  for (t in 2:T)
    beta[t] ~ normal(beta[t-1], sigma);
  sigma ~ lognormal(-3, 1);

  // 尤度
  for (n in 1:N) {
    vector[K] cprob;
    for (k in 1:K) {
      real s = 0;
      for (m in 1:k) {
        s += theta[jj[n]] - b[ii[n]] - beta[tt[n]] - d[m];
      }
      cprob[k] = s;
    }
    target += cprob[y[n]] - log_sum_exp(cprob);
  }
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    vector[K] cprob;
    for (k in 1:K) {
      real s = 0;
      for (m in 1:k) {
        s += theta[jj[n]] - b[ii[n]] - beta[tt[n]] - d[m];
      }
      cprob[k] = s;
    }
    log_lik[n] = cprob[y[n]] - log_sum_exp(cprob);
  }
}