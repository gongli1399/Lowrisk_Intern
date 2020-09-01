import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.stats import norm
from scipy import optimize


class OptionPricing:

    def __init__(self, intr, divr, sigma):

        '''
        Args:
            intr: interest rate
            divr: dividend rate
            sigma: sigma (volatility)
        '''

        self.intr = intr
        self.divr = divr
        self.sigma = sigma

    def BsmModel(self, strike, spot, texp, cp_sign):

        '''
        Black-Scholes-Merton model for option pricing.

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp_sign: 1 for call or -1 for put option

        Returns:
            Vanilla option price and d1

        Example:
            model = OptionPricing(intr=0.06, divr=0, sigma=0.2)
            price = model.BsmModel(strike=100, spot=100, texp=1, cp_sign=1)[0]
        '''

        disc_fac = np.exp(-texp * self.intr)
        fwd = spot * np.exp(-texp * self.divr) / disc_fac

        sigma_std = self.sigma * np.sqrt(texp)
        d1 = np.log(fwd / strike) / sigma_std + 0.5 * sigma_std
        d2 = d1 - sigma_std

        bsm_price = cp_sign * disc_fac * (fwd * norm.cdf(cp_sign * d1) - strike * norm.cdf(cp_sign * d2))

        return bsm_price, d1

    def implicitMethod(self, strike, spot, texp, cp_sign, style, N, M):

        '''
        Implicit method for option pricing.

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp_sign: 1 for call or -1 for put option
            style: 1 for European or 0 for American option
            N: time steps (N = int(10000*texp) is suggested)
            M: space steps (M = int(20*np.sqrt(N)) is suggested)

        Returns:
            Price for European or American option

        Example:
            model = OptionPricing(intr=0.06, divr=0, sigma=0.2)
            texp = 1
            N = int(10000*texp)
            M = int(20*np.sqrt(N))
            price = model.implicitMethod(strike=100, spot=100, texp=texp, cp_sign=-1, style=0, N=N, M=M)
        '''

        t, dt = np.linspace(0, texp, N + 1, retstep=True)
        dx = self.sigma * np.sqrt(dt)
        x_min = np.minimum(np.log(strike) - M * dx / 2, np.log(spot))
        x_max = np.maximum(np.log(strike) + M * dx / 2, np.log(spot))
        M = int((x_max - x_min) / dx) + 1
        x_max = x_min + M * dx
        x = np.linspace(x_min, x_max, M + 1)

        dxx = dx * dx
        sigma2 = self.sigma * self.sigma
        a = dt / 2 * ((self.intr - self.divr - 0.5 * sigma2) / dx - sigma2 / dxx)
        b = 1 + dt * (self.intr + sigma2 / dxx)
        c = -dt / 2 * ((self.intr - self.divr - 0.5 * sigma2) / dx + sigma2 / dxx)

        terminal = np.maximum(cp_sign * (np.exp(x[1:-1]) - strike), 0)
        bound_min = np.zeros(N + 1) if cp_sign == 1 else strike * np.exp(-self.intr * style * (texp - t)) - np.exp(
            x_min - self.divr * style * (texp - t))
        bound_max = np.exp(x_max - self.divr * style * (texp - t)) - strike * np.exp(
            -self.intr * style * (texp - t)) if cp_sign == 1 else np.zeros(N + 1)

        A = sparse.diags([a, b, c], [-1, 0, 1], shape=(M - 1, M - 1))
        A_LU = splu(A.tocsc())
        B = np.zeros(M - 1)
        F = np.maximum(cp_sign * (np.exp(x[1:-1]) - strike), 0)

        if style == 1:
            for i in range(N - 1, -1, -1):
                B[0] = a * bound_min[i]
                B[-1] = c * bound_max[i]
                F = A_LU.solve(F - B)
        elif style == 0:
            for i in range(N - 1, -1, -1):
                B[0] = a * bound_min[i]
                B[-1] = c * bound_max[i]
                F = np.maximum(A_LU.solve(F - B), terminal)

        return np.interp(np.log(spot), x[1:-1], F)

    def explicitMethod(self, strike, spot, texp, cp_sign, style, N, M):

        '''
        Explicit method for option pricing.

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp_sign: 1 for call or -1 for put option
            style: 1 for European or 0 for American option
            N: time steps (N = int(10000*texp) is suggested)
            M: space steps (M = int(20*np.sqrt(N)) is suggested)

        Returns:
            Price for European or American option

        Example:
            model = OptionPricing(intr=0.06, divr=0, sigma=0.2)
            texp = 1
            N = int(10000*texp)
            M = int(20*np.sqrt(N))
            price = model.explicitMethod(strike=100, spot=100, texp=texp, cp_sign=-1, style=0, N=N, M=M)
        '''

        t, dt = np.linspace(0, texp, N + 1, retstep=True)
        dx = self.sigma * np.sqrt(dt)
        x_min = np.minimum(np.log(strike) - M * dx / 2, np.log(spot))
        x_max = np.maximum(np.log(strike) + M * dx / 2, np.log(spot))
        M = int((x_max - x_min) / dx) + 1
        x_max = x_min + M * dx
        x = np.linspace(x_min, x_max, M + 1)

        dxx = dx * dx
        sigma2 = self.sigma * self.sigma
        disc_fac = 1 / (1 + self.intr * dt)
        a = disc_fac * dt / 2 * (sigma2 / dxx - (self.intr - self.divr - 0.5 * sigma2) / dx)
        b = disc_fac * (1 - sigma2 * dt / dxx)
        c = disc_fac * dt / 2 * (sigma2 / dxx + (self.intr - self.divr - 0.5 * sigma2) / dx)

        terminal = np.maximum(cp_sign * (np.exp(x[1:-1]) - strike), 0)
        bound_min = np.zeros(N + 1) if cp_sign == 1 else strike * np.exp(-self.intr * style * (texp - t)) - np.exp(
            x_min - self.divr * style * (texp - t))
        bound_max = np.exp(x_max - self.divr * style * (texp - t)) - strike * np.exp(
            -self.intr * style * (texp - t)) if cp_sign == 1 else np.zeros(N + 1)
        F = np.maximum(cp_sign * (np.exp(x) - strike), 0)

        if style == 1:
            for i in range(N - 1, -1, -1):
                F[1:-1] = a * F[:-2] + b * F[1:-1] + c * F[2:]
                F[0] = bound_min[i]
                F[-1] = bound_max[i]
        elif style == 0:
            for i in range(N - 1, -1, -1):
                F[1:-1] = np.maximum(a * F[:-2] + b * F[1:-1] + c * F[2:], terminal)
                F[0] = bound_min[i]
                F[-1] = bound_max[i]

        return np.interp(np.log(spot), x, F)

    def CrankNicolson(self, strike, spot, texp, cp_sign, style, N, M):

        '''
        Crank-Nicolson method for option pricing.

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp_sign: 1 for call or -1 for put option
            style: 1 for European or 0 for American option
            N: time steps (N = int(10000*texp) is suggested)
            M: space steps (M = int(20*np.sqrt(N)) is suggested)

        Returns:
            Price for European or American option

        Example:
            model = OptionPricing(intr=0.06, divr=0, sigma=0.2)
            texp = 1
            N = int(10000*texp)
            M = int(20*np.sqrt(N))
            price = model.CrankNicolson(strike=100, spot=100, texp=texp, cp_sign=-1, style=0, N=N, M=M)
        '''

        t, dt = np.linspace(0, texp, N + 1, retstep=True)
        dx = self.sigma * np.sqrt(dt)
        x_min = np.minimum(np.log(strike) - M * dx / 2, np.log(spot))
        x_max = np.maximum(np.log(strike) + M * dx / 2, np.log(spot))
        M = int((x_max - x_min) / dx) + 1
        x_max = x_min + M * dx
        x = np.linspace(x_min, x_max, M + 1)

        dxx = dx * dx
        sigma2 = self.sigma * self.sigma
        disc_fac = 1 / (1 + self.intr * dt)

        a_im = dt / 2 * ((self.intr - self.divr - 0.5 * sigma2) / dx - sigma2 / dxx)
        b_im = 1 + dt * (self.intr + sigma2 / dxx)
        c_im = -dt / 2 * ((self.intr - self.divr - 0.5 * sigma2) / dx + sigma2 / dxx)

        a_ex = disc_fac * dt / 2 * (sigma2 / dxx - (self.intr - self.divr - 0.5 * sigma2) / dx)
        b_ex = disc_fac * (1 - sigma2 * dt / dxx)
        c_ex = disc_fac * dt / 2 * (sigma2 / dxx + (self.intr - self.divr - 0.5 * sigma2) / dx)

        terminal = np.maximum(cp_sign * (np.exp(x[1:-1]) - strike), 0)
        bound_min = np.zeros(N + 1) if cp_sign == 1 else strike * np.exp(-self.intr * style * (texp - t)) - np.exp(
            x_min - self.divr * style * (texp - t))
        bound_max = np.exp(x_max - self.divr * style * (texp - t)) - strike * np.exp(
            -self.intr * style * (texp - t)) if cp_sign == 1 else np.zeros(N + 1)

        A_im = sparse.diags([a_im, b_im + 1, c_im], [-1, 0, 1], shape=(M - 1, M - 1))
        A_im_LU = splu(A_im.tocsc())
        B = np.zeros(M - 1)

        A_ex = sparse.diags([a_ex, b_ex + 1, c_ex], [-1, 0, 1], shape=(M - 1, M - 1))
        F = np.maximum(cp_sign * (np.exp(x[1:-1]) - strike), 0)

        if style == 1:
            for i in range(N - 1, -1, -1):
                B[0] = a_ex * bound_min[i + 1] - a_im * bound_min[i]
                B[-1] = c_ex * bound_max[i + 1] - c_im * bound_max[i]
                F = A_im_LU.solve(A_ex @ F + B)
        elif style == 0:
            for i in range(N - 1, -1, -1):
                B[0] = a_ex * bound_min[i + 1] - a_im * bound_min[i]
                B[-1] = c_ex * bound_max[i + 1] - c_im * bound_max[i]
                F = np.maximum(A_im_LU.solve(A_ex @ F + B), terminal)

        return np.interp(np.log(spot), x[1:-1], F)

    def binomial_tree(self, strike, spot, texp, cp_sign, style, N=10000):

        '''
        Binomial options pricing model.

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp_sign: 1 for call or -1 for put option
            style: 1 for European or 0 for American option
            N: time steps (N = int(10000*texp) is suggested)

        Returns:
            Price for European or American option

        Example:
            model = OptionPricing(intr=0.06, divr=0, sigma=0.2)
            texp = 1
            N = int(10000*texp)
            price = model.binomial_tree(strike=100, spot=100, texp=texp, cp_sign=-1, style=0, N=N)
        '''

        dt = texp / N

        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((self.intr - self.divr) * dt) - d) / (u - d)
        q = 1 - p

        disc_fac = np.exp(-self.intr * dt)

        S = spot * u ** np.arange(N + 1) * d ** np.arange(N, -1, -1)
        V = np.maximum(cp_sign * (S - strike), 0)

        if style == 1:
            for i in np.arange(N, 0, -1):
                V = (p * V[1:] + (1 - p) * V[:-1]) * disc_fac
        elif style == 0:
            for i in np.arange(N, 0, -1):
                V = (p * V[1:] + (1 - p) * V[:-1]) * disc_fac
                S = S[:-1] * u
                V = np.maximum(V, cp_sign * (S - strike))

        return V[0]

    def LongstaffSchwartz(self, strike, spot, texp, cp_sign, n_paths=10000, n_steps=1000):

        '''
        Valuing American Options by Simulation: A Simple Least-Squares Approach

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp_sign: 1 for call or -1 for put option
            n_paths: number of simulation paths
            n_steps: time steps

        Returns:
            Estimate and standard error of American option price

        Example:
            model = OptionPricing(intr=0.06, divr=0, sigma=0.2)
            estimate, std_error = model.LongstaffSchwartz(strike=100, spot=100, texp=1, cp_sign=-1, n_paths=10000, n_steps=1000)
        '''

        dt = texp / n_steps
        disc_fac = np.exp(-self.intr * dt)

        stock_paths = np.ones((n_paths, n_steps + 1))
        zz = np.random.normal(size=(int(n_paths / 2), n_steps))
        zz = np.concatenate((zz, -zz), axis=0)

        stock_paths[:, 1:] = np.cumprod(
            np.exp((self.intr - self.divr - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * zz), axis=1)
        stock_paths = stock_paths * spot

        V = np.maximum(cp_sign * (stock_paths[:, -1] - strike), 0)

        for i in range(n_steps - 1, -1, -1):

            exercise_value = np.maximum(cp_sign * (stock_paths[:, i] - strike), 0)
            select_paths = exercise_value > 0

            if np.sum(select_paths) >= 3:
                ols = np.polyfit(stock_paths[select_paths, i], V[select_paths] * disc_fac, 2)
                expected_hold_values = np.polyval(ols, stock_paths[select_paths, i])

                exercise_paths = np.zeros(n_paths, dtype=bool)
                exercise_paths[select_paths] = exercise_value[select_paths] >= expected_hold_values

                V[~exercise_paths] = V[~exercise_paths] * disc_fac
                V[exercise_paths] = exercise_value[exercise_paths]

            else:
                V = V * disc_fac

        return np.mean(V), np.std(V) / np.sqrt(n_paths)

    def exercise_boundary_func(self, strike, spot, texp, cp_sign):

        bsm_price, d1 = self.BsmModel(strike, spot, texp, cp_sign)

        X = 1 - np.exp(-texp * self.intr)
        sigma2 = self.sigma * self.sigma
        L = 2 * (self.intr - self.divr) / sigma2
        M = 2 * self.intr / sigma2
        q = (-(L - 1) + cp_sign * np.sqrt((L - 1) ** 2 + 4 * M / X)) / 2

        A = cp_sign * (1 - np.exp(-texp * self.divr) * norm.cdf(cp_sign * d1)) * spot / q

        bound_func = cp_sign * (spot - strike) - bsm_price - A

        return bound_func

    def critical_value(self, strike, texp, cp_sign):

        '''
        Analytic approximation of optimal exercise boundary for American option.

        Args:
            strike: strike price
            texp: time to expiry
            cp_sign: 1 for call or -1 for put option

        Returns:
            Optimal exercise boundary for American option

        Example:
            model = OptionPricing(intr=0.06, divr=0, sigma=0.2)
            critical_value = model.critical_value(strike=100, texp=1, cp_sign=-1)

        Warning:
            CANNOT give optimal exercise boundary for American call on non-dividend paying stock since American call on
            non-dividend paying stock will never exercised early and the optimal exercise boundary will be infinite. Raise error!
        '''

        bound_func = lambda _spot: self.exercise_boundary_func(strike, _spot, texp, cp_sign)

        return optimize.brentq(bound_func, 0.01, 10 * strike)

    def BAWModel(self, strike, spot, texp, cp_sign):

        '''
        Efficient Analytic Approximation of American Option Values.

        Args:
            strike: strike price
            spot: spot price
            texp: time to expiry
            cp_sign: 1 for call or -1 for put option

        Returns:
            American option price

        Example:
            model = OptionPricing(intr=0.06, divr=0, sigma=0.2)
            price = model.BAWModel(strike=100, spot=100, texp=1, cp_sign=-1)

        Warning:
            CANNOT value American call on non-dividend paying stock since American call on non-dividend paying
            stock will never exercised early and the optimal exercise boundary will be infinite. Raise error!
        '''

        bsm_price = self.BsmModel(strike, spot, texp, cp_sign)[0]
        critical_value = self.critical_value(strike, texp, cp_sign)

        sigma_std = self.sigma * np.sqrt(texp)
        d1 = (np.log(critical_value / strike) + (self.intr - self.divr) * texp) / sigma_std + 0.5 * sigma_std

        X = 1 - np.exp(-texp * self.intr)
        sigma2 = self.sigma * self.sigma
        L = 2 * (self.intr - self.divr) / sigma2
        M = 2 * self.intr / sigma2
        q = (-(L - 1) + cp_sign * np.sqrt((L - 1) ** 2 + 4 * M / X)) / 2

        A = cp_sign * (1 - np.exp(-texp * self.divr) * norm.cdf(cp_sign * d1)) * critical_value / q

        BAW_price = bsm_price + A * (spot / critical_value) ** q if cp_sign * (
                    spot - critical_value) < 0 else cp_sign * (spot - strike)

        return BAW_price