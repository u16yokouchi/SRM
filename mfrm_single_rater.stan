data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> J_free;
  int<lower=1> T;
  int<lower=1> K;
  int<lower=1,upper=J> jj[N];
  int<lower=1,upper=T> tt[N];
  int<lower=1,upper=K> y[N];
  int<lower=1> n_fixed;
  int<lower=1,upper=J> fixed_j[n_fixed];
  int<lower=0,upper=1> is_fixed[J];
}
parameters {
  vector[J] theta;
  vector[T] beta;
  vector[K] d;
  real<lower=0> sigma;
}
model {
  // θのうち0固定すべきところを強制的に0にする
  for (j in 1:J) {
    if (is_fixed[j] == 1) {
      target += normal_lpdf(theta[j] | 0, 1e-8); // ほぼ0に固定
    } else {
      target += normal_lpdf(theta[j] | 0, 1);
    }
  }
  d ~ normal(0, 1);
  beta[1] ~ normal(0, 1);
  for (t in 2:T)
    beta[t] ~ normal(beta[t-1], sigma);
  sigma ~ lognormal(-3, 1);

  for (n in 1:N) {
    vector[K] cprob;
    for (k in 1:K) {
      real s = 0;
      for (m in 1:k)
        s += theta[jj[n]] - beta[tt[n]] - d[m];
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
      for (m in 1:k)
        s += theta[jj[n]] - beta[tt[n]] - d[m];
      cprob[k] = s;
    }
    log_lik[n] = cprob[y[n]] - log_sum_exp(cprob);
  }
}
