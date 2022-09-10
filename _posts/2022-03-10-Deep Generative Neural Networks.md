---
title: Deep Generative Neural Networks
author: Weizhi Peng
date: 2022-03-10 18:32:00 -0500
categories: [Deep Learning, Deep Generative Neural Networks]
tags: [Deep Learning, Deep Generative Neural Networks]
---


![Screen Shot 2022-05-13 at 12.46.29.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_12.46.29.png)

# Maximum Likelihood

![Screen Shot 2022-05-13 at 12.50.25.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_12.50.25.png)

# Generate image

![Screen Shot 2022-05-13 at 12.52.08.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_12.52.08.png)

![Screen Shot 2022-05-13 at 12.52.33.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_12.52.33.png)

![Screen Shot 2022-05-13 at 12.53.12.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_12.53.12.png)

![Screen Shot 2022-05-13 at 12.53.32.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_12.53.32.png)

![Screen Shot 2022-05-13 at 12.54.59.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_12.54.59.png)

# ****Generative Adversarial Networks****

![Screen Shot 2022-05-13 at 12.59.00.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_12.59.00.png)

![Screen Shot 2022-05-13 at 12.59.20.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_12.59.20.png)

![Screen Shot 2022-05-13 at 13.00.55.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.00.55.png)

![Screen Shot 2022-05-13 at 13.01.40.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.01.40.png)

![Screen Shot 2022-05-13 at 13.02.42.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.02.42.png)

![Screen Shot 2022-05-13 at 13.03.13.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.03.13.png)

![Screen Shot 2022-05-13 at 13.03.47.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.03.47.png)

![Screen Shot 2022-05-13 at 13.17.49.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.17.49.png)

![Screen Shot 2022-05-13 at 13.18.23.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.18.23.png)

![Screen Shot 2022-05-13 at 13.19.17.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.19.17.png)


### Generative Adversarial Networks python code

    # calculating cost function
    import numpy as np
    X = [
        [1,2],
        [3,4]
    ]
    X = np.array(X)
    X_pred = [
        [5,6],
        [7,8]
    ]
    X_pred = np.array(X_pred)

    weight = np.ones(2)*0.5

    # define of Discriminator
    def Dx(x,a = 0.1,b = 0.2):
        x = np.transpose(x)
        return 1/(1+np.exp(-a*x[0]+b*x[1]+2))

    # Calculate the cost
    def V(x,x_pred = X_pred ,weight = weight,Dx = Dx):
        # E(ln(D(x)))
        E_lnDx = np.dot(np.log(Dx(X)),weight.transpose())
        # E(ln(1-D(G(x)))),G(x) = X_pred
        E_ln_1_DGx = np.dot(np.log(1-Dx(X_pred)),weight.transpose())
        return E_lnDx + E_ln_1_DGx


    print(V(X,X_pred,weight,Dx))

Output

    -2.5465207684425333

Continue Python code

    from jax import grad
    import jax.numpy as jnp
    import numpy as np
    a = [0.1,0.2]
    def Dm_1(a,x = X, b = 0.2):
        temp = 1/(jnp.exp(jnp.dot(jnp.array([-a,b]),jnp.transpose(x)) + 2)+1)
        return temp
        

    def Dm_2(a,x = X[0], b = 0.1):
        return 1/(jnp.exp(jnp.dot(jnp.array([-b,a]),jnp.transpose(x)) + 2)+1)


    def multi_Dm_1(a,x):
        result = []
        for sample in x:
            temp_func = grad(Dm_1)(a,sample)
            result.append(temp_func)
        return np.array(result)

    def multi_Dm_2(a,x):
        result = []
        for sample in x:
            temp_func = grad(Dm_2)(a,sample)
            result.append(temp_func)
        return np.array(result)


    print(1/Dx(X)*multi_Dm_1(0.1,X))

Output

    [0.90887703 2.77242559]

Continue Python code

    # Calculate gradient update
    from jax import grad
    import jax.numpy as jnp
    import numpy as np
    X = [
        [1,2],
        [3,4]
    ]
    X = np.array(X)
    X_pred = [
        [5,6],
        [7,8]
    ]
    X_pred = np.array(X_pred)
    weight = np.ones(2)*0.5

    # define learning rate
    n = 0.02

    a = [0.1,0.2]

    def Dm_1(a,x = X, b = 0.2):
        temp = 1/(jnp.exp(jnp.dot(jnp.array([-a,b]),jnp.transpose(x)) + 2)+1)
        return temp
        

    def Dm_2(a,x = X[0], b = 0.1):
        return 1/(jnp.exp(jnp.dot(jnp.array([-b,a]),jnp.transpose(x)) + 2)+1)


    def multi_Dm_1(a,x):
        result = []
        for sample in x:
            temp_func = grad(Dm_1)(a,sample)
            result.append(temp_func)
        return np.array(result)

    def multi_Dm_2(a,x):
        result = []
        for sample in x:
            temp_func = grad(Dm_2)(a,sample)
            result.append(temp_func)
        return np.array(result)

    def d_V(x,x_pred = X_pred ,weight = weight,Dx = Dx):
        # d(ln(D(x)))/dm1 + d(ln(1 - D(G(x))))/dm1 = d(ln(D(x))/d(D(x)) * d(D(x))/dm1+ 
        # d(ln(1 - D(G(x))))/d(1 - D(G(x))) * d(1 - D(G(x)))/d(D(G(x))) * d(D(G(x)))/dm1
        d_m1 = 1/Dx(x) * multi_Dm_1(a[0],X) + 1/(1-Dx(x_pred)) * (-1) * multi_Dm_1(a[0],x_pred)
        
        # d(ln(D(x)))/dm2 + d(ln(1 - D(G(x))))/dm2 = d(ln(D(x))/d(D(x)) * d(D(x))/dm2+ 
        # d(ln(1 - D(G(x))))/d(1 - D(G(x))) * d(1 - D(G(x)))/d(D(G(x))) * d(D(G(x)))/dm2
        d_m2 = 1/Dx(x) * multi_Dm_2(a[1],X) + 1/(1-Dx(x_pred)) * (-1) * multi_Dm_2(a[1],x_pred)

        # sum and apply weight
        result = np.dot([d_m1,d_m2],weight)
        return result

    update = d_V(X)
    a_new = a + n*update
    print(a_new)


Output

    [0.13001361 0.15280747]


# GAN Problems

![Screen Shot 2022-05-13 at 13.20.50.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.20.50.png)

### None-convergence

![Screen Shot 2022-05-13 at 13.23.04.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.23.04.png)

### Diminished gradient

![Screen Shot 2022-05-13 at 13.24.11.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.24.11.png)

![Screen Shot 2022-05-13 at 13.25.14.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.25.14.png)

![Screen Shot 2022-05-13 at 13.25.36.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.25.36.png)

![Screen Shot 2022-05-13 at 13.25.55.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.25.55.png)

![Screen Shot 2022-05-13 at 13.26.33.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.26.33.png)

![Screen Shot 2022-05-13 at 13.26.47.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.26.47.png)

![Screen Shot 2022-05-13 at 13.27.03.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.27.03.png)

![Screen Shot 2022-05-13 at 13.27.19.png](Week%206%20Deep%20Generative%20Neural%20Networks%201722c20af05d4b04b0bca3f3f396be18/Screen_Shot_2022-05-13_at_13.27.19.png)