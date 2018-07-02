data {
  int <lower=1> T;
  vector[T] y;
}
transformed data {
  real sd_y = sd(y);
}
parameters {
  real<lower=0> sigma_y;
  vector[T] gamma;
  vector[T] zeta;
  real eta;
  real<lower=-1, upper=1> phi;
}
transformed parameters {
  vector[T] u;
  vector[T] v;

  u[1] = y[1] + gamma[1];
  v[1] = zeta[1];
  for (t in 2:T) {
    u[t] = u[t-1] + v[t-1] + gamma[t-1];
    v[t] = eta + phi * (v[t-1] - eta) + zeta[t-1];
  }
}
model {
  // likelihood
  y ~ normal(u, sigma_y);

  // priors
  sigma_y ~ exponential(1 / sd_y);
  gamma ~ student_t(3, 0, 2.5);
  zeta ~ student_t(3, 0, 1);
  eta ~ student_t(3, 0, 1);
  phi ~ normal(0, 0.5);
}
generated quantities {
  vector[T] y_pred;
  for (t in 1:T)
    y_pred[t] = normal_rng(u[t], sigma_y);
}

