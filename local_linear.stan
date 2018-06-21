data {
  int <lower=1> T;
  vector[T] y;
}
transformed data {
  real sd_y = sd(y);
}
parameters {
  real<lower=0> y_err;
  vector[T] u_err;
  real<lower=0> u_tau;
  vector[T] v_err;
  real<lower=0> v_tau;
  real eta;
  real<lower=0> eta_tau;
  real phi;
}
transformed parameters {
  vector[T] u;
  vector[T] v;

  u[1] = y[1] + u_tau * u_err[1];
  v[1] = v_err[1];
  for (t in 2:T) {
    u[t] = u[t-1] + v[t-1] + u_tau * u_err[t-1];
    v[t] = eta + phi*(v[t-1] - eta) + v_tau * v_err[t-1];
  }
}
model {
  // likelihood
  y ~ normal(u, y_err);

  // priors
  y_err ~ gamma(2, 1 / sd_y);
  u_err ~ normal(0, 1);
  u_tau ~ gamma(2, 1);
  v_err ~ normal(0, 1);
  v_tau ~ gamma(2, 10);
  eta ~ normal(0, eta_tau);
  eta_tau ~ student_t(3, 0, 2.5);
  phi ~ normal(0, 0.5);
}
generated quantities {
  vector[T] y_pred;
  for (t in 1:T)
    y_pred[t] = normal_rng(u[t], y_err);
}
