---
title: Discriminant Functions
author: Weizhi Peng
date: 2022-02-01 18:32:00 -0500
categories: [Deep Learning, Discriminant Functions]
tags: [Deep Learning, Discriminant Functions, python, code, tutorial]
---


# Deep Learning: Discriminant Functions
Here is my [Deep Learning Full Tutorial](https://pengwz.info/categories/deep-learning/)!

Discriminant functions divide feature space into regions

## Linear Discriminant Functions

![Screen Shot 2022-05-10 at 15.38.14.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-10_at_15.38.14.png)

![Screen Shot 2022-05-10 at 15.40.05.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-10_at_15.40.05.png)

![Screen Shot 2022-05-10 at 15.40.17.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-10_at_15.40.17.png)

![Screen Shot 2022-05-10 at 15.41.19.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-10_at_15.41.19.png)

![Screen Shot 2022-05-10 at 15.42.41.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-10_at_15.42.41.png)


### Generalised Linear Discriminant Functions

![Screen Shot 2022-05-10 at 15.43.32.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-10_at_15.43.32.png)

### Dichotomizer Python code

    import numpy as np
    X = [
        [1,1],
        [2,2],
        [3,3]
    ]
    w = [2,1]
    b = -5
    def g(w,x,b):
        return np.dot(w,x) + b

    result = [g(w,x,b) for x in X]
    print(result)

Output

    [-2, 1, 4]

## Learning Decision Boundaries

## perceptron learning

Perceptron Criterion Function

![Screen Shot 2022-05-10 at 16.02.43.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-10_at_16.02.43.png)

![Screen Shot 2022-05-10 at 16.03.03.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-10_at_16.03.03.png)

![Screen Shot 2022-05-11 at 22.19.43.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-11_at_22.19.43.png)

# Sequential Perceptron Learning Algorithm

![Screen Shot 2022-05-11 at 22.51.24.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-11_at_22.51.24.png)

![Screen Shot 2022-05-11 at 22.51.40.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-11_at_22.51.40.png)

![Screen Shot 2022-05-11 at 22.53.58.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-11_at_22.53.58.png)

![Screen Shot 2022-05-11 at 22.54.20.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-11_at_22.54.20.png)

### Sequential Perceptron Learning Algorithm Python code

    # Sequential Perceptron Learning Algorithm
    import numpy as np
    from prettytable import PrettyTable

    # configuration variables
    # -----------------------------------------------------------
    # initial values
    # a = [b, w1, w2]
    a = [-1.5,5,-1]

    # learning rate
    n = 1
    epoch = 3
    # dataset
    # -----------------------------------------------------------
    X = [[0,0],[1,0],[2,1],[0,1],[1,2]]
    Y = [1,1,1,-1,-1]

    # Sequential Perceptron Learning Algorithm
    # -----------------------------------------------------------
    result = []
    for o in range(epoch):
        for i in range(len(Y)):
            a_prev = a
            # y = [1, x1, x2]
            y = np.hstack((1,X[i]))
            # calculate ay like wx+b = b + w1x1 + w2x2
            ay = np.dot(a, y)

            #update part ηω'yk
            if ay * Y[i] < 0:
                if ay > 0:
                    update = n * y * (-1)
                if ay < 0:
                    update = n * y 
                # add update part to a
                a = np.add(a, update)

            cur_result = []
            # evaluate
            for index in range(len(Y)):
                cur_result.append((np.dot(np.hstack((1,X[index])),a)))

            # check if converage
            is_converage = True
            for index in range(len(Y)):
                if cur_result[index] * Y[index] <= 0:
                    is_converage = False

                
            # append result
            result.append((str(i + 1 + (len(Y) * o)), np.round(a_prev, 4), np.round(y, 4), np.round(ay, 4), np.round(a, 4),np.round(cur_result, 4),is_converage))

    # prettytable
    # -----------------------------------------------------------
    pt = PrettyTable(('iteration', 'a', 'y', 'ay', 'a_new','over all result','is converage'))
    for row in result: pt.add_row(row)
    print(pt)

Output

    +-----------+------------------+---------+------+------------------+----------------------------+-------------+
    | iteration | a                | y       | ay   | a_new            |      over all result       | is converge |
    +-----------+------------------+---------+------+------------------+----------------------------+-------------+
    |     1     | [-1.5  5.  -1. ] | [1 0 0] | -1.5 | [-0.5  5.  -1. ] | [-0.5  4.5  8.5 -1.5  2.5] |    False    |
    |     2     | [-0.5  5.  -1. ] | [1 1 0] | 4.5  | [-0.5  5.  -1. ] | [-0.5  4.5  8.5 -1.5  2.5] |    False    |
    |     3     | [-0.5  5.  -1. ] | [1 2 1] | 8.5  | [-0.5  5.  -1. ] | [-0.5  4.5  8.5 -1.5  2.5] |    False    |
    |     4     | [-0.5  5.  -1. ] | [1 0 1] | -1.5 | [-0.5  5.  -1. ] | [-0.5  4.5  8.5 -1.5  2.5] |    False    |
    |     5     | [-0.5  5.  -1. ] | [1 1 2] | 2.5  | [-1.5  4.  -3. ] | [-1.5  2.5  3.5 -4.5 -3.5] |    False    |
    |     6     | [-1.5  4.  -3. ] | [1 0 0] | -1.5 | [-0.5  4.  -3. ] | [-0.5  3.5  4.5 -3.5 -2.5] |    False    |
    |     7     | [-0.5  4.  -3. ] | [1 1 0] | 3.5  | [-0.5  4.  -3. ] | [-0.5  3.5  4.5 -3.5 -2.5] |    False    |
    |     8     | [-0.5  4.  -3. ] | [1 2 1] | 4.5  | [-0.5  4.  -3. ] | [-0.5  3.5  4.5 -3.5 -2.5] |    False    |
    |     9     | [-0.5  4.  -3. ] | [1 0 1] | -3.5 | [-0.5  4.  -3. ] | [-0.5  3.5  4.5 -3.5 -2.5] |    False    |
    |     10    | [-0.5  4.  -3. ] | [1 1 2] | -2.5 | [-0.5  4.  -3. ] | [-0.5  3.5  4.5 -3.5 -2.5] |    False    |
    |     11    | [-0.5  4.  -3. ] | [1 0 0] | -0.5 | [ 0.5  4.  -3. ] | [ 0.5  4.5  5.5 -2.5 -1.5] |     True    |
    |     12    | [ 0.5  4.  -3. ] | [1 1 0] | 4.5  | [ 0.5  4.  -3. ] | [ 0.5  4.5  5.5 -2.5 -1.5] |     True    |
    |     13    | [ 0.5  4.  -3. ] | [1 2 1] | 5.5  | [ 0.5  4.  -3. ] | [ 0.5  4.5  5.5 -2.5 -1.5] |     True    |
    |     14    | [ 0.5  4.  -3. ] | [1 0 1] | -2.5 | [ 0.5  4.  -3. ] | [ 0.5  4.5  5.5 -2.5 -1.5] |     True    |
    |     15    | [ 0.5  4.  -3. ] | [1 1 2] | -1.5 | [ 0.5  4.  -3. ] | [ 0.5  4.5  5.5 -2.5 -1.5] |     True    |
    +-----------+------------------+---------+------+------------------+----------------------------+-------------+

### Batch Perceptron Learning Algorithm

    # Batch Perceptron Learning Algorithm
    import numpy as np
    from prettytable import PrettyTable

    # configuration variables
    # -----------------------------------------------------------
    # initial values
    # a = [b, w1, w2]
    a = [-25,6,3]

    # learning rate
    n = 1
    epoch = 3
    # dataset
    # -----------------------------------------------------------
    X = [[1,5],[2,5],[4,1],[5,1]]
    Y = [1,1,-1,-1]

    # Sequential Perceptron Learning Algorithm
    # -----------------------------------------------------------
    result = []
    for o in range(epoch):
        update = 0
        for i in range(len(Y)):
            a_prev = a
            # y = [1, x1, x2]
            y = np.hstack((1,X[i]))
            # calculate ay like wx+b = b + w1x1 + w2x2
            ay = np.dot(a, y)

            #update part ηω'yk
            if ay * Y[i] <= 0:
                if ay > 0:
                    update = update + n * y * (-1)
                if ay <= 0:
                    update = update + n * y 

            cur_result = []
            # evaluate
            for index in range(len(Y)):
                cur_result.append((np.dot(np.hstack((1,X[index])),a)))

            # check if converage
            is_converage = True
            for index in range(len(Y)):
                if cur_result[index] * Y[index] <= 0:
                    is_converage = False

                
            # append result
            result.append((str(i + 1 + (len(Y) * o)), np.round(a_prev, 4), np.round(y, 4), np.round(ay, 4), np.round(a, 4),np.round(cur_result, 4),is_converage))
        # add update part to a
        a = np.add(a, update)

    # prettytable
    # -----------------------------------------------------------
    pt = PrettyTable(('iteration', 'a', 'y', 'ay', 'a_new','over all result','is converage'))
    for row in result: pt.add_row(row)
    print(pt)

Output

    +-----------+---------------+---------+-----+---------------+-------------------+--------------+
    | iteration |       a       |    y    |  ay |     a_new     |  over all result  | is converage |
    +-----------+---------------+---------+-----+---------------+-------------------+--------------+
    |     1     | [-25   6   3] | [1 1 5] |  -4 | [-25   6   3] |   [-4  2  2  8]   |    False     |
    |     2     | [-25   6   3] | [1 2 5] |  2  | [-25   6   3] |   [-4  2  2  8]   |    False     |
    |     3     | [-25   6   3] | [1 4 1] |  2  | [-25   6   3] |   [-4  2  2  8]   |    False     |
    |     4     | [-25   6   3] | [1 5 1] |  8  | [-25   6   3] |   [-4  2  2  8]   |    False     |
    |     5     | [-26  -2   6] | [1 1 5] |  2  | [-26  -2   6] | [  2   0 -28 -30] |    False     |
    |     6     | [-26  -2   6] | [1 2 5] |  0  | [-26  -2   6] | [  2   0 -28 -30] |    False     |
    |     7     | [-26  -2   6] | [1 4 1] | -28 | [-26  -2   6] | [  2   0 -28 -30] |    False     |
    |     8     | [-26  -2   6] | [1 5 1] | -30 | [-26  -2   6] | [  2   0 -28 -30] |    False     |
    |     9     | [-25   0  11] | [1 1 5] |  30 | [-25   0  11] | [ 30  30 -14 -14] |     True     |
    |     10    | [-25   0  11] | [1 2 5] |  30 | [-25   0  11] | [ 30  30 -14 -14] |     True     |
    |     11    | [-25   0  11] | [1 4 1] | -14 | [-25   0  11] | [ 30  30 -14 -14] |     True     |
    |     12    | [-25   0  11] | [1 5 1] | -14 | [-25   0  11] | [ 30  30 -14 -14] |     True     |
    +-----------+---------------+---------+-----+---------------+-------------------+--------------+

### Sequential Multiclass Perceptron Learning algorithm Python code

    # Sequential Multiclass Perceptron Learning algorithm
    import numpy as np
    from prettytable import PrettyTable

    # configuration variables
    # -----------------------------------------------------------
    # initial values
    # a = [b, w1, w2]
    a = [[-0.5,0,-0.5],[-3,0.5,0],[0.5,1.5,-0.5]]

    # learning rate
    n = 1
    epoch = 5
    # dataset
    # -----------------------------------------------------------
    X = [[0,1],[1,0],[0.5,1.5],[1,1],[-0.5,0]]
    Y = [1,1,2,2,3]

    # Sequential Perceptron Learning Algorithm
    # -----------------------------------------------------------
    result = []
    for o in range(epoch):
        for i in range(len(Y)):

            a_prev = a.copy()
            # y = [1, x1, x2]
            y = np.hstack((1,X[i]))
            # calculate ay like wx+b = b + w1x1 + w2x2
            ay = np.dot(a, np.transpose([y]))
            if all(x == ay[0] for x in ay):
                j = 2
            else:
                j = np.argmax(ay)

            true_y = Y[i] - 1

            if j != true_y:
                update =  n * y 
                a[true_y] = np.add(a[true_y], update)
                update =  n * y * (-1)
                a[j] = np.add(a[j], update)

            cur_result = []
            # evaluate
            for index in range(len(Y)):
                cur_result.append((np.dot(np.hstack((1,X[index])),np.transpose(a))))
            # check if converage
            is_converage = False
            if np.array_equal(np.array(Y),np.argmax(cur_result,axis=1) + 1):
                is_converage = True

                
            # append result
            result.append((str(i + 1 + (len(Y) * o)), np.round(a_prev, 4), np.round(y, 4), np.round(ay, 4),j+1, np.round(a, 4),np.round(cur_result, 4),np.argmax(cur_result,axis=1) + 1,is_converage))

    # prettytable
    # -----------------------------------------------------------
    pt = PrettyTable(('iteration', 'a', 'y', 'ay','class', 'a_new','over all result','class result','is converage'))
    for row in result: pt.add_row(row)
    print(pt)

Output

    +-----------+--------------------+------------------+-----------+-------+--------------------+-----------------------+--------------+--------------+
    | iteration |         a          |        y         |     ay    | class |       a_new        |    over all result    | class result | is converage |
    +-----------+--------------------+------------------+-----------+-------+--------------------+-----------------------+--------------+--------------+
    |     1     | [[-0.5  0.  -0.5]  |     [1 0 1]      |   [[-1.]  |   3   | [[ 0.5  0.   0.5]  |  [[ 1.   -3.   -2.  ] | [1 3 1 1 1]  |    False     |
    |           |  [-3.   0.5  0. ]  |                  |    [-3.]  |       |  [-3.   0.5  0. ]  |   [ 0.5  -2.5   1.  ] |              |              |
    |           |  [ 0.5  1.5 -0.5]] |                  |   [ 0.]]  |       |  [-0.5  1.5 -1.5]] |   [ 1.25 -2.75 -2.  ] |              |              |
    |           |                    |                  |           |       |                    |   [ 1.   -2.5  -0.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [ 0.5  -3.25 -1.25]] |              |              |
    |     2     | [[ 0.5  0.   0.5]  |     [1 1 0]      |  [[ 0.5]  |   3   | [[ 1.5  1.   0.5]  |  [[ 2.   -3.   -3.  ] | [1 1 1 1 1]  |    False     |
    |           |  [-3.   0.5  0. ]  |                  |   [-2.5]  |       |  [-3.   0.5  0. ]  |   [ 2.5  -2.5  -1.  ] |              |              |
    |           |  [-0.5  1.5 -1.5]] |                  |   [ 1. ]] |       |  [-1.5  0.5 -1.5]] |   [ 2.75 -2.75 -3.5 ] |              |              |
    |           |                    |                  |           |       |                    |   [ 3.   -2.5  -2.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [ 1.   -3.25 -1.75]] |              |              |
    |     3     | [[ 1.5  1.   0.5]  |  [1.  0.5 1.5]   |  [[ 2.75] |   1   | [[ 0.5  0.5 -1. ]  |  [[-0.5  -0.5  -3.  ] | [1 1 2 2 1]  |    False     |
    |           |  [-3.   0.5  0. ]  |                  |   [-2.75] |       |  [-2.   1.   1.5]  |   [ 1.   -1.   -1.  ] |              |              |
    |           |  [-1.5  0.5 -1.5]] |                  |  [-3.5 ]] |       |  [-1.5  0.5 -1.5]] |   [-0.75  0.75 -3.5 ] |              |              |
    |           |                    |                  |           |       |                    |   [ 0.    0.5  -2.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [ 0.25 -2.5  -1.75]] |              |              |
    |     4     | [[ 0.5  0.5 -1. ]  |     [1 1 1]      |  [[ 0. ]  |   2   | [[ 0.5  0.5 -1. ]  |  [[-0.5  -0.5  -3.  ] | [1 1 2 2 1]  |    False     |
    |           |  [-2.   1.   1.5]  |                  |   [ 0.5]  |       |  [-2.   1.   1.5]  |   [ 1.   -1.   -1.  ] |              |              |
    |           |  [-1.5  0.5 -1.5]] |                  |   [-2.5]] |       |  [-1.5  0.5 -1.5]] |   [-0.75  0.75 -3.5 ] |              |              |
    |           |                    |                  |           |       |                    |   [ 0.    0.5  -2.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [ 0.25 -2.5  -1.75]] |              |              |
    |     5     | [[ 0.5  0.5 -1. ]  | [ 1.  -0.5  0. ] |  [[ 0.25] |   1   | [[-0.5  1.  -1. ]  |  [[-1.5  -0.5  -2.  ] | [2 1 2 2 3]  |    False     |
    |           |  [-2.   1.   1.5]  |                  |   [-2.5 ] |       |  [-2.   1.   1.5]  |   [ 0.5  -1.   -0.5 ] |              |              |
    |           |  [-1.5  0.5 -1.5]] |                  |  [-1.75]] |       |  [-0.5  0.  -1.5]] |   [-1.5   0.75 -2.75] |              |              |
    |           |                    |                  |           |       |                    |   [-0.5   0.5  -2.  ] |              |              |
    |           |                    |                  |           |       |                    |  [-1.   -2.5  -0.5 ]] |              |              |
    |     6     | [[-0.5  1.  -1. ]  |     [1 0 1]      |  [[-1.5]  |   2   | [[ 0.5  1.   0. ]  |  [[ 0.5  -2.5  -2.  ] | [1 1 1 1 1]  |    False     |
    |           |  [-2.   1.   1.5]  |                  |   [-0.5]  |       |  [-3.   1.   0.5]  |   [ 1.5  -2.   -0.5 ] |              |              |
    |           |  [-0.5  0.  -1.5]] |                  |   [-2. ]] |       |  [-0.5  0.  -1.5]] |   [ 1.   -1.75 -2.75] |              |              |
    |           |                    |                  |           |       |                    |   [ 1.5  -1.5  -2.  ] |              |              |
    |           |                    |                  |           |       |                    |  [ 0.   -3.5  -0.5 ]] |              |              |
    |     7     | [[ 0.5  1.   0. ]  |     [1 1 0]      |  [[ 1.5]  |   1   | [[ 0.5  1.   0. ]  |  [[ 0.5  -2.5  -2.  ] | [1 1 1 1 1]  |    False     |
    |           |  [-3.   1.   0.5]  |                  |   [-2. ]  |       |  [-3.   1.   0.5]  |   [ 1.5  -2.   -0.5 ] |              |              |
    |           |  [-0.5  0.  -1.5]] |                  |   [-0.5]] |       |  [-0.5  0.  -1.5]] |   [ 1.   -1.75 -2.75] |              |              |
    |           |                    |                  |           |       |                    |   [ 1.5  -1.5  -2.  ] |              |              |
    |           |                    |                  |           |       |                    |  [ 0.   -3.5  -0.5 ]] |              |              |
    |     8     | [[ 0.5  1.   0. ]  |  [1.  0.5 1.5]   |  [[ 1.  ] |   1   | [[-0.5  0.5 -1.5]  |  [[-2.    0.   -2.  ] | [2 1 2 2 3]  |    False     |
    |           |  [-3.   1.   0.5]  |                  |   [-1.75] |       |  [-2.   1.5  2. ]  |   [ 0.   -0.5  -0.5 ] |              |              |
    |           |  [-0.5  0.  -1.5]] |                  |  [-2.75]] |       |  [-0.5  0.  -1.5]] |   [-2.5   1.75 -2.75] |              |              |
    |           |                    |                  |           |       |                    |   [-1.5   1.5  -2.  ] |              |              |
    |           |                    |                  |           |       |                    |  [-0.75 -2.75 -0.5 ]] |              |              |
    |     9     | [[-0.5  0.5 -1.5]  |     [1 1 1]      |  [[-1.5]  |   2   | [[-0.5  0.5 -1.5]  |  [[-2.    0.   -2.  ] | [2 1 2 2 3]  |    False     |
    |           |  [-2.   1.5  2. ]  |                  |   [ 1.5]  |       |  [-2.   1.5  2. ]  |   [ 0.   -0.5  -0.5 ] |              |              |
    |           |  [-0.5  0.  -1.5]] |                  |   [-2. ]] |       |  [-0.5  0.  -1.5]] |   [-2.5   1.75 -2.75] |              |              |
    |           |                    |                  |           |       |                    |   [-1.5   1.5  -2.  ] |              |              |
    |           |                    |                  |           |       |                    |  [-0.75 -2.75 -0.5 ]] |              |              |
    |     10    | [[-0.5  0.5 -1.5]  | [ 1.  -0.5  0. ] |  [[-0.75] |   3   | [[-0.5  0.5 -1.5]  |  [[-2.    0.   -2.  ] | [2 1 2 2 3]  |    False     |
    |           |  [-2.   1.5  2. ]  |                  |   [-2.75] |       |  [-2.   1.5  2. ]  |   [ 0.   -0.5  -0.5 ] |              |              |
    |           |  [-0.5  0.  -1.5]] |                  |  [-0.5 ]] |       |  [-0.5  0.  -1.5]] |   [-2.5   1.75 -2.75] |              |              |
    |           |                    |                  |           |       |                    |   [-1.5   1.5  -2.  ] |              |              |
    |           |                    |                  |           |       |                    |  [-0.75 -2.75 -0.5 ]] |              |              |
    |     11    | [[-0.5  0.5 -1.5]  |     [1 0 1]      |   [[-2.]  |   2   | [[ 0.5  0.5 -0.5]  |  [[ 0.   -2.   -2.  ] | [1 1 1 1 1]  |    False     |
    |           |  [-2.   1.5  2. ]  |                  |    [ 0.]  |       |  [-3.   1.5  1. ]  |   [ 1.   -1.5  -0.5 ] |              |              |
    |           |  [-0.5  0.  -1.5]] |                  |   [-2.]]  |       |  [-0.5  0.  -1.5]] |   [ 0.   -0.75 -2.75] |              |              |
    |           |                    |                  |           |       |                    |   [ 0.5  -0.5  -2.  ] |              |              |
    |           |                    |                  |           |       |                    |  [ 0.25 -3.75 -0.5 ]] |              |              |
    |     12    | [[ 0.5  0.5 -0.5]  |     [1 1 0]      |  [[ 1. ]  |   1   | [[ 0.5  0.5 -0.5]  |  [[ 0.   -2.   -2.  ] | [1 1 1 1 1]  |    False     |
    |           |  [-3.   1.5  1. ]  |                  |   [-1.5]  |       |  [-3.   1.5  1. ]  |   [ 1.   -1.5  -0.5 ] |              |              |
    |           |  [-0.5  0.  -1.5]] |                  |   [-0.5]] |       |  [-0.5  0.  -1.5]] |   [ 0.   -0.75 -2.75] |              |              |
    |           |                    |                  |           |       |                    |   [ 0.5  -0.5  -2.  ] |              |              |
    |           |                    |                  |           |       |                    |  [ 0.25 -3.75 -0.5 ]] |              |              |
    |     13    | [[ 0.5  0.5 -0.5]  |  [1.  0.5 1.5]   |  [[ 0.  ] |   1   | [[-0.5  0.  -2. ]  |  [[-2.5   0.5  -2.  ] | [2 2 2 2 1]  |    False     |
    |           |  [-3.   1.5  1. ]  |                  |   [-0.75] |       |  [-2.   2.   2.5]  |   [-0.5   0.   -0.5 ] |              |              |
    |           |  [-0.5  0.  -1.5]] |                  |  [-2.75]] |       |  [-0.5  0.  -1.5]] |   [-3.5   2.75 -2.75] |              |              |
    |           |                    |                  |           |       |                    |   [-2.5   2.5  -2.  ] |              |              |
    |           |                    |                  |           |       |                    |  [-0.5  -3.   -0.5 ]] |              |              |
    |     14    | [[-0.5  0.  -2. ]  |     [1 1 1]      |  [[-2.5]  |   2   | [[-0.5  0.  -2. ]  |  [[-2.5   0.5  -2.  ] | [2 2 2 2 1]  |    False     |
    |           |  [-2.   2.   2.5]  |                  |   [ 2.5]  |       |  [-2.   2.   2.5]  |   [-0.5   0.   -0.5 ] |              |              |
    |           |  [-0.5  0.  -1.5]] |                  |   [-2. ]] |       |  [-0.5  0.  -1.5]] |   [-3.5   2.75 -2.75] |              |              |
    |           |                    |                  |           |       |                    |   [-2.5   2.5  -2.  ] |              |              |
    |           |                    |                  |           |       |                    |  [-0.5  -3.   -0.5 ]] |              |              |
    |     15    | [[-0.5  0.  -2. ]  | [ 1.  -0.5  0. ] |  [[-0.5]  |   1   | [[-1.5  0.5 -2. ]  |  [[-3.5   0.5  -1.  ] | [2 2 2 2 3]  |    False     |
    |           |  [-2.   2.   2.5]  |                  |   [-3. ]  |       |  [-2.   2.   2.5]  |   [-1.    0.    0.  ] |              |              |
    |           |  [-0.5  0.  -1.5]] |                  |   [-0.5]] |       |  [ 0.5 -0.5 -1.5]] |   [-4.25  2.75 -2.  ] |              |              |
    |           |                    |                  |           |       |                    |   [-3.    2.5  -1.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [-1.75 -3.    0.75]] |              |              |
    |     16    | [[-1.5  0.5 -2. ]  |     [1 0 1]      |  [[-3.5]  |   2   | [[-0.5  0.5 -1. ]  |  [[-1.5  -1.5  -1.  ] | [3 1 2 2 3]  |    False     |
    |           |  [-2.   2.   2.5]  |                  |   [ 0.5]  |       |  [-3.   2.   1.5]  |   [ 0.   -1.    0.  ] |              |              |
    |           |  [ 0.5 -0.5 -1.5]] |                  |   [-1. ]] |       |  [ 0.5 -0.5 -1.5]] |   [-1.75  0.25 -2.  ] |              |              |
    |           |                    |                  |           |       |                    |   [-1.    0.5  -1.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [-0.75 -4.    0.75]] |              |              |
    |     17    | [[-0.5  0.5 -1. ]  |     [1 1 0]      |   [[ 0.]  |   1   | [[-0.5  0.5 -1. ]  |  [[-1.5  -1.5  -1.  ] | [3 1 2 2 3]  |    False     |
    |           |  [-3.   2.   1.5]  |                  |    [-1.]  |       |  [-3.   2.   1.5]  |   [ 0.   -1.    0.  ] |              |              |
    |           |  [ 0.5 -0.5 -1.5]] |                  |   [ 0.]]  |       |  [ 0.5 -0.5 -1.5]] |   [-1.75  0.25 -2.  ] |              |              |
    |           |                    |                  |           |       |                    |   [-1.    0.5  -1.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [-0.75 -4.    0.75]] |              |              |
    |     18    | [[-0.5  0.5 -1. ]  |  [1.  0.5 1.5]   |  [[-1.75] |   2   | [[-0.5  0.5 -1. ]  |  [[-1.5  -1.5  -1.  ] | [3 1 2 2 3]  |    False     |
    |           |  [-3.   2.   1.5]  |                  |   [ 0.25] |       |  [-3.   2.   1.5]  |   [ 0.   -1.    0.  ] |              |              |
    |           |  [ 0.5 -0.5 -1.5]] |                  |  [-2.  ]] |       |  [ 0.5 -0.5 -1.5]] |   [-1.75  0.25 -2.  ] |              |              |
    |           |                    |                  |           |       |                    |   [-1.    0.5  -1.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [-0.75 -4.    0.75]] |              |              |
    |     19    | [[-0.5  0.5 -1. ]  |     [1 1 1]      |  [[-1. ]  |   2   | [[-0.5  0.5 -1. ]  |  [[-1.5  -1.5  -1.  ] | [3 1 2 2 3]  |    False     |
    |           |  [-3.   2.   1.5]  |                  |   [ 0.5]  |       |  [-3.   2.   1.5]  |   [ 0.   -1.    0.  ] |              |              |
    |           |  [ 0.5 -0.5 -1.5]] |                  |   [-1.5]] |       |  [ 0.5 -0.5 -1.5]] |   [-1.75  0.25 -2.  ] |              |              |
    |           |                    |                  |           |       |                    |   [-1.    0.5  -1.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [-0.75 -4.    0.75]] |              |              |
    |     20    | [[-0.5  0.5 -1. ]  | [ 1.  -0.5  0. ] |  [[-0.75] |   3   | [[-0.5  0.5 -1. ]  |  [[-1.5  -1.5  -1.  ] | [3 1 2 2 3]  |    False     |
    |           |  [-3.   2.   1.5]  |                  |   [-4.  ] |       |  [-3.   2.   1.5]  |   [ 0.   -1.    0.  ] |              |              |
    |           |  [ 0.5 -0.5 -1.5]] |                  |  [ 0.75]] |       |  [ 0.5 -0.5 -1.5]] |   [-1.75  0.25 -2.  ] |              |              |
    |           |                    |                  |           |       |                    |   [-1.    0.5  -1.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [-0.75 -4.    0.75]] |              |              |
    |     21    | [[-0.5  0.5 -1. ]  |     [1 0 1]      |  [[-1.5]  |   3   | [[ 0.5  0.5  0. ]  |  [[ 0.5  -1.5  -3.  ] | [1 1 1 1 1]  |    False     |
    |           |  [-3.   2.   1.5]  |                  |   [-1.5]  |       |  [-3.   2.   1.5]  |   [ 1.   -1.   -1.  ] |              |              |
    |           |  [ 0.5 -0.5 -1.5]] |                  |   [-1. ]] |       |  [-0.5 -0.5 -2.5]] |   [ 0.75  0.25 -4.5 ] |              |              |
    |           |                    |                  |           |       |                    |   [ 1.    0.5  -3.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [ 0.25 -4.   -0.25]] |              |              |
    |     22    | [[ 0.5  0.5  0. ]  |     [1 1 0]      |   [[ 1.]  |   1   | [[ 0.5  0.5  0. ]  |  [[ 0.5  -1.5  -3.  ] | [1 1 1 1 1]  |    False     |
    |           |  [-3.   2.   1.5]  |                  |    [-1.]  |       |  [-3.   2.   1.5]  |   [ 1.   -1.   -1.  ] |              |              |
    |           |  [-0.5 -0.5 -2.5]] |                  |   [-1.]]  |       |  [-0.5 -0.5 -2.5]] |   [ 0.75  0.25 -4.5 ] |              |              |
    |           |                    |                  |           |       |                    |   [ 1.    0.5  -3.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [ 0.25 -4.   -0.25]] |              |              |
    |     23    | [[ 0.5  0.5  0. ]  |  [1.  0.5 1.5]   |  [[ 0.75] |   1   | [[-0.5  0.  -1.5]  |  [[-2.    1.   -3.  ] | [2 2 2 2 3]  |    False     |
    |           |  [-3.   2.   1.5]  |                  |   [ 0.25] |       |  [-2.   2.5  3. ]  |   [-0.5   0.5  -1.  ] |              |              |
    |           |  [-0.5 -0.5 -2.5]] |                  |  [-4.5 ]] |       |  [-0.5 -0.5 -2.5]] |   [-2.75  3.75 -4.5 ] |              |              |
    |           |                    |                  |           |       |                    |   [-2.    3.5  -3.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [-0.5  -3.25 -0.25]] |              |              |
    |     24    | [[-0.5  0.  -1.5]  |     [1 1 1]      |  [[-2. ]  |   2   | [[-0.5  0.  -1.5]  |  [[-2.    1.   -3.  ] | [2 2 2 2 3]  |    False     |
    |           |  [-2.   2.5  3. ]  |                  |   [ 3.5]  |       |  [-2.   2.5  3. ]  |   [-0.5   0.5  -1.  ] |              |              |
    |           |  [-0.5 -0.5 -2.5]] |                  |   [-3.5]] |       |  [-0.5 -0.5 -2.5]] |   [-2.75  3.75 -4.5 ] |              |              |
    |           |                    |                  |           |       |                    |   [-2.    3.5  -3.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [-0.5  -3.25 -0.25]] |              |              |
    |     25    | [[-0.5  0.  -1.5]  | [ 1.  -0.5  0. ] |  [[-0.5 ] |   3   | [[-0.5  0.  -1.5]  |  [[-2.    1.   -3.  ] | [2 2 2 2 3]  |    False     |
    |           |  [-2.   2.5  3. ]  |                  |   [-3.25] |       |  [-2.   2.5  3. ]  |   [-0.5   0.5  -1.  ] |              |              |
    |           |  [-0.5 -0.5 -2.5]] |                  |  [-0.25]] |       |  [-0.5 -0.5 -2.5]] |   [-2.75  3.75 -4.5 ] |              |              |
    |           |                    |                  |           |       |                    |   [-2.    3.5  -3.5 ] |              |              |
    |           |                    |                  |           |       |                    |  [-0.5  -3.25 -0.25]] |              |              |
    +-----------+--------------------+------------------+-----------+-------+--------------------+-----------------------+--------------+--------------+


## Minimum Squared Error (MSE) Procedures

![Screen Shot 2022-05-11 at 23.54.07.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-11_at_23.54.07.png)

![Screen Shot 2022-05-12 at 00.07.13.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-12_at_00.07.13.png)

![Screen Shot 2022-05-12 at 00.07.22.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-12_at_00.07.22.png)

### Minimum Squared Error (MSE) Procedures Python code

    # Typically Y is rectangular, so the solution is given by: a=Y† b
    b = np.dot(y, a)

    # Cost function
    # e2= ∥ Ya−b ∥^2, J s (a)= 1/2‖ Ya−b ‖^2
    e = (np.dot(y, a)-b)(np.dot(y, a)-b)


# Widrow-Hoff (or LMS) Learning Algorithm

![Screen Shot 2022-05-12 at 00.07.53.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-12_at_00.07.53.png)

![Screen Shot 2022-05-12 at 00.08.09.png](https://raw.githubusercontent.com/wzptech/wzptech.github.io/main/assets/post-images/Week%202%20Discriminant%20Functions%20b0ab70fc0a964f04ab3e4e6405ef9a32/Screen_Shot_2022-05-12_at_00.08.09.png)


### Sequential Widrow-Hoff Learning Algorithm Python code

    # Sequential Widrow-Hoff Learning Algorithm
    import numpy as np
    from prettytable import PrettyTable

    # configuration variables
    # -----------------------------------------------------------
    # initial values
    a = [-1.5,5,-1]
    # learning rate
    n = 0.2

    # iterations
    # iterations = 150
    epoch = 2
    # dataset
    # -----------------------------------------------------------
    X = [[0,0],[1,0],[2,1],[0,1],[1,2]]
    Y = [1,1,1,-1,-1]
    # margin vector
    b = np.ones(len(X))*2


    # sequential widrow-hoff learning algorithm
    # -----------------------------------------------------------
    result = []
    for o in range(epoch):
        for i in range(len(Y)):
            a_prev = a
            if Y[i] == 1:
                y = np.hstack((Y[i],X[i]))
            else:
                y = np.hstack((Y[i],np.array(X[i])*(-1)))

            # calculate ay
            ay = np.dot(a, y)
            

            # calculate update part, η(b−ay)*y
            update = n * (b[i] - ay) * y

            # add update part to a
            a = np.add(a, update)

            cur_result = []
            # evaluate
            for index in range(len(Y)):
                y = np.hstack((1,X[index]))
                cur_result.append(np.dot(a, y))

            # check if converage
            is_converage = True
            for index in range(len(Y)):
                if cur_result[index] * Y[index] < 0:
                    is_converage = False

            # append result
            result.append((str(i + 1 + (len(Y) * o)), np.round(a_prev, 4), np.round(y, 4), np.round(ay, 4), np.round(a, 4),np.round(cur_result, 4),is_converage))

    # prettytable
    # -----------------------------------------------------------
    pt = PrettyTable(('iteration', 'a', 'y', 'ay', 'a_new','over all result','is converage'))
    for row in result: pt.add_row(row)
    print(pt)

Output

    +-----------+---------------------------+---------+---------+---------------------------+-------------------------------------------+-------------+
    | iteration | a                         | y       | ay      | a_new                     |              over all result              | is converge |
    +-----------+---------------------------+---------+---------+---------------------------+-------------------------------------------+-------------+
    |     1     | [-1.5  5.  -1. ]          | [1 1 2] | -1.5    | [-0.8  5.  -1. ]          |         [-0.8  4.2  8.2 -1.8  2.2]        |    False    |
    |     2     | [-0.8  5.  -1. ]          | [1 1 2] | 4.2     | [-1.24  4.56 -1.  ]       |      [-1.24  3.32  6.88 -2.24  1.32]      |    False    |
    |     3     | [-1.24  4.56 -1.  ]       | [1 1 2] | 6.88    | [-2.216  2.608 -1.976]    |    [-2.216  0.392  1.024 -4.192 -3.56 ]   |    False    |
    |     4     | [-2.216  2.608 -1.976]    | [1 1 2] | 4.192   | [-1.7776  2.608  -1.5376] | [-1.7776  0.8304  1.9008 -3.3152 -2.2448] |    False    |
    |     5     | [-1.7776  2.608  -1.5376] | [1 1 2] | 2.2448  | [-1.7286  2.657  -1.4397] | [-1.7286  0.9283  2.1456 -3.1683 -1.951 ] |    False    |
    |     6     | [-1.7286  2.657  -1.4397] | [1 1 2] | -1.7286 | [-0.9829  2.657  -1.4397] | [-0.9829  1.674   2.8913 -2.4226 -1.2053] |    False    |
    |     7     | [-0.9829  2.657  -1.4397] | [1 1 2] | 1.674   | [-0.9177  2.7222 -1.4397] | [-0.9177  1.8044  3.0869 -2.3574 -1.0749] |    False    |
    |     8     | [-0.9177  2.7222 -1.4397] | [1 1 2] | 3.0869  | [-1.1351  2.2874 -1.6571] | [-1.1351  1.1523  1.7826 -2.7922 -2.1618] |    False    |
    |     9     | [-1.1351  2.2874 -1.6571] | [1 1 2] | 2.7922  | [-0.9767  2.2874 -1.4986] | [-0.9767  1.3107  2.0995 -2.4753 -1.6865] |    False    |
    |     10    | [-0.9767  2.2874 -1.4986] | [1 1 2] | 1.6865  | [-1.0394  2.2247 -1.624 ] | [-1.0394  1.1853  1.786  -2.6634 -2.0627] |    False    |
    +-----------+---------------------------+---------+---------+---------------------------+-------------------------------------------+-------------+