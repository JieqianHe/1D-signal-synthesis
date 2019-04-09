Scattering + Generator

Model:
x -> Scattering -> S(x) -> Generator -> G(S(x)) -> Scattering -> S(G(S(x)))

Loss:
MSE between S(x) and S(G(S(x)))

Goal:
aim to find G that generates signals whose scattering coefficients matches that of real signals

Currently can do up to 2 layers where s2 / s1 is a constant.
