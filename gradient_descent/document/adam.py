def adam(x_start, step, g):             #Adaptive Moment Estimation
    x = np.array(x_start, dtype='float64')
    beta1=0.9
    beta2=0.999
    mt = np.zeros_like(x)
    vt = np.zeros_like(x)
    dp = np.zeros_like(x)
    passing_dot = [x.copy()]
    for i in range(n_iter):
        beta1t=beta1**(i+1)
        beta2t=beta2**(i+1)
        temp=g(x)
        mt=beta1*mt + (1-beta1) * g(x)
        vt=beta2*vt + (1-beta2) * g(x)**2
        mt1=mt/(1-beta1t)
        vt1=vt/(1-beta2t)
        dp[0]= step * mt1[0]/(math.sqrt(vt1[0])+0.00000001)
        dp[1] = step * mt1[1] / (math.sqrt(vt1[1]) + 0.00000001)
        x -= dp
        passing_dot.append(x.copy())
        if abs(sum(mt1)) < 1e-6:
            break;
    return x, passing_dot


