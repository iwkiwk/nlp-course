import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

titanic_content = pd.read_csv('titanic_train.csv')

titanic_content.head(10)

titanic_content = titanic_content.dropna()
titanic_content.head(10)

age_with_fare = titanic_content[['Age', 'Fare']]
age_with_fare = age_with_fare[
    (age_with_fare['Age'] > 22) & (age_with_fare['Fare'] < 400) & (age_with_fare['Fare'] > 130)]

age = np.array(age_with_fare['Age'])
fare = np.array(age_with_fare['Fare'])

plt.scatter(age, fare)


def loss(y_true, yhats):
    return np.mean(np.abs(y_true - yhats))


def model(x, a, b):
    return a * x + b


a = 100
b = 100


def yhat_func(a, b):
    return np.array([model(x, a, b) for x in age])


eps = 20

directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]

learning_rate = 1e-2

min_loss = float('inf')

batch = 0
total = 10000

while True:
    if min_loss < eps:
        print('a: {}, b: {}, min_loss: {}'.format(a, b, min_loss))
        break

    indices = np.random.choice(range(len(age)), size=10)

    sample_x = age[indices]
    sample_y = fare[indices]

    new_a, new_b = a, b

    current_loss = float('inf')
    for d in directions:
        da, db = d

        if min_loss != float('inf'):
            _a = a + da * min_loss * learning_rate
            _b = b + db * min_loss * learning_rate
        else:
            _a, _b = a + da, b + db

        y_hats = [model(x, _a, _b) for x in sample_x]
        current_loss = loss(sample_y, np.array([model(x, a + da, b + db) for x in sample_x]))

        if current_loss < min_loss:
            min_loss = current_loss
            new_a, new_b = _a, _b
            print('batch {}, fare: {} * age + {}, min_loss: {}'.format(batch, new_a, new_b, min_loss))

    if batch > total:
        break

    batch += 1

    a, b = new_a, new_b

    # time.sleep(0.002)

print("all samples' loss: {}".format(loss(age, yhat_func(a, b))))
plt.plot(age, [model(x, a, b) for x in age])
plt.show()
