"""Koenies et al. (2018) ITPA TAE: zero-orbit-width growth rate vs fast-ion temperature."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

import fax

s = jnp.linspace(1e-3, 1.0, 201)
r = jnp.sqrt(s)
c0, c1, c2, c3 = 0.49123, 0.298228, 0.198739, 0.521298
density = 1.44131e17 * c3 * jnp.exp(-(c2 / c1) * jnp.tanh((r - c0) / c2))
mode = jnp.exp(-(((r - 0.5) / 0.0425) ** 2) / 2)
phi = jnp.stack([mode, mode], axis=1)  # model TAE: m = 10, 11 centred on q = 1.75
plasma = fax.Plasma(q=1.71 + 0.16 * s)
temps = jnp.linspace(50.0, 500.0, 19)
gamma = [fax.growth_rate(4.0e5, phi, s, ((10, 6), (11, 6)), plasma, fax.FastIons(density, t * 1e3)) for t in temps]
plt.plot(temps, jnp.array(gamma) / 1e3, "k-", label="FAX (ZOW, passing)")
plt.errorbar([100, 200, 300, 400], [9, 29, 40.5, 46.5], yerr=[1, 4, 2.5, 1.5], fmt="o", label="Koenies 2018 ZLR band")
plt.xlabel("T_f (keV)")
plt.ylabel("gamma (10^3 / s)")
plt.legend()
plt.savefig("koenies_tae_drive.png", dpi=120)
