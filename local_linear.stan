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
  vector[T] v_err;
  real eta;
  real<lower=-1, upper=1> phi;
}
transformed parameters {
  vector[T] u;
  vector[T] v;

  u[1] = y[1] + u_err[1];
  v[1] = v_err[1];
  for (t in 2:T) {
    u[t] = u[t-1] + v[t-1] + u_err[t-1];
    v[t] = eta + phi * (v[t-1] - eta) + v_err[t-1];
  }
}
model {
  // likelihood
  y ~ normal(u, y_err);

  // priors
  y_err ~ exponential(1 / sd_y);
  u_err ~ student_t(3, 0, 1);
  v_err ~ student_t(3, 0, 1);
  eta ~ student_t(3, 0, 1);
  phi ~ normal(0, 0.5);
}
generated quantities {
  vector[T] y_pred;
  for (t in 1:T)
    y_pred[t] = normal_rng(u[t], y_err);
}

