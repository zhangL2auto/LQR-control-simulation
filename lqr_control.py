'''
Author: zhangL
Email: deutschlei47@126.com
Date: 2023-06-08 09:01:14
Version: 0.0.1
LastEditors: zhangL2auto deutschlei47@126.com
LastEditTime: 2023-06-08 13:08:43
FilePath: /3. month/pso/lqr_self.py
Description:  
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy


def solve_lqr(A, B, Q, R):
    # 状态转移矩阵
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    # LQR增益矩阵
    K = scipy.linalg.inv(B.T.dot(P).dot(B) + R).dot(B.T.dot(P).dot(A))
    # 特征值
    E, _ = np.linalg.eig(A - B.dot(K))

    return P, K, E

def sys():
    A = np.array([[1,1], [0,1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0], [0, 1]])

    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1]])
    return A, B, Q, R

def do_lqr(A, B, Q, R):
    P, K, E = solve_lqr(A, B, Q, R)
    # 初始状态和步长
    x0 = np.array([[2], [1]])
    dt = 0.01
    # 仿真参数
    t = np.arange(0, 10, dt)
    x = np.ones((2, len(t)))*5
    u = np.zeros((1, len(t)))

    x_ref = np.array([np.sin(t), np.sin(t)*6])
    x_delta = x - x_ref
    x_delta[:, 0] = x0[:, 0]

    for i in range(len(t)-1):
        u[:, i] = -K @ x_delta[:, i]
        x_delta[:, i+1] = A @ x_delta[:, i] + B @ u[:, i]

    x = x_delta + x_ref
    return t, x, x_ref

def plot_x_u(t, x, x_ref):
    plt.figure("out")
    for i in range(len(t)-1):
        plt.cla()
        plt.plot(t, x_ref[0, :], "-.b", label = 'ref1')
        plt.plot(t, x_ref[1, :], "-.r", label = 'ref2')
        plt.plot(t[:i], x[0, :i], label='path1')
        plt.plot(t[:i], x[1, :i], label='path2')
        plt.scatter(t[i], x[0, i], marker="v", label='auto1')
        plt.scatter(t[i], x[1, i], marker="s", label='auto2')
        plt.legend()
        plt.pause(0.001)

plt.show()

def main():
    A, B, Q, R = sys()
    t, x ,x_ref = do_lqr(A, B, Q, R)
    plot_x_u(t, x, x_ref)

if __name__ == '__main__':
    main()
