data {
  int <lower=1> T;
  vector[T] y;
  real y_scale;
  real u_scale;
  real v_scale;
}
transformed data {
  real sd_y = sd(y);
}
parameters {
  vector[T] u_err;
  real<lower=0> u_tau;
  vector[T] v_err;
  real<lower=0> v_tau;
  real<lower=0> y_err;
}
transformed parameters {
  vector[T] u;
  vector[T] v;

  u[1] = y[1] + u_tau * u_err[1];
  v[1] = v_tau * v_err[1];
  for (t in 2:T) {
    u[t] = u[t-1] + v[t-1] + u_tau * u_err[t-1];
    v[t] = v[t-1] + v_err[t-1];
  }
}
model {
  // priors
  v_err ~ normal(0, 1);
  v_tau ~ gamma(2, 1 / v_scale);
  u_err ~ normal(0, 1);
  u_tau ~ gamma(2, 1 / u_scale);
  y_err ~ gamma(2, 1 / (y_scale * sd_y));

  // likelihood
  y ~ normal(u, y_err);
}
generated quantities {
  vector[T] y_pred;
  for (t in 1:T)
    y_pred[t] = normal_rng(u[t], y_err);
}
