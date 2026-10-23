"""Koenies et al. (2018) ITPA TAE: zero-orbit-width growth rate vs fast-ion temperature."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

import fax

s = (jnp.arange(200) + 0.5) / 200
r = jnp.sqrt(s)
c0, c1, c2, c3 = 0.49123, 0.298228, 0.198739, 0.521298
density = 1.44131e17 * c3 * jnp.exp(-(c2 / c1) * jnp.tanh((r - c0) / c2))
mode = jnp.exp(-(((r - 0.5) / 0.0425) ** 2) / 2)
phi = jnp.stack([mode, mode], axis=1)  # model TAE: m = 10, 11 centred on q = 1.75
geo = fax.circular_torus(s=s, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), epsilon_B=0.1)
rho = fax.mass_density(2e19, 1.00727647)
temps = jnp.linspace(50.0, 500.0, 19)
gamma = [fax.growth_rate(4.0e5, phi, geo, ((10, 6), (11, 6)), fax.FastIons(density, t * 1e3), rho, helical=0, nzeta=1) for t in temps]
plt.plot(temps, jnp.array(gamma) / 1e3, "k-", label="FAX (ZOW, passing)")
plt.errorbar([100, 200, 300, 400], [9, 29, 40.5, 46.5], yerr=[1, 4, 2.5, 1.5], fmt="o", label="Koenies 2018 ZLR band")
plt.xlabel("T_f (keV)")
plt.ylabel("gamma (10^3 / s)")
plt.legend()
plt.savefig("koenies_tae_drive.png", dpi=120)
