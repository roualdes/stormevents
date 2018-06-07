data {
  int <lower=0> T;
  vector[T] y;
}
transformed data {
  real sd_y = sd(y);
}
parameters {
  vector[T] u_err;
  real<lower=0> u_tau;
  vector[T] v_err;
  real<lower=0> y_err;
}
transformed parameters {
  vector[T] u;
  vector[T] v;

  u[1] = y[1] + u_tau * u_err[1];
  v[1] = v_err[1];
  for (t in 2:T) {
    u[t] = u[t-1] + v[t-1] + u_tau * u_err[t-1];
    v[t] = v[t-1] + v_err[t-1];
  }
}
model {
  // priors
  v_err ~ normal(0, 1);
  u_err ~ normal(0, 1);
  u_tau ~ normal(0, 10);
  y_err ~ exponential(1 / sd_y);

  // likelihood
  y ~ normal(u, y_err);
}
generated quantities {
  vector[T] y_pred;
  for (t in 1:T)
    y_pred[t] = normal_rng(u[t], y_err);
}
