import time
import numpy as np
from MF import fun
import copy
import matplotlib.pyplot as plt
import multiprocessing
if __name__ == "__main__":
    cpu_num=4
    tic = time.time()
    c1 = 1.49445
    c2 = 1.49445
    maxg = 20 # 迭代次数
    sizepop = 20 # 种群规模

    paranum = 2 # 参数个数
    yy = np.ones([maxg,1])
    # Vmax=1
    # Vmin=-1
    popmax = [2 ,10]
    popmin = [1 ,2]
    pop = np.ones([sizepop, paranum])
    # print(pop)
    V = np.ones([sizepop, paranum])
    # print(V)
    fitness = np.ones(sizepop)
    for i in range(sizepop):
        for j in range(paranum):
            pop[i,j] = np.random.random()*(popmax[j]-popmin[j])+popmin[j]

        # pop[i,:]= np.random.random([1,paranum])
            V[i,:] = np.random.random([1,paranum])

    with multiprocessing.Pool(processes=cpu_num) as pool:
        fitness = pool.map(fun, pop)
    print(fitness)

    # pool = multiprocessing.Pool(processes=cpu_num)
    # fitness = pool.map(fun, pop)
    # print(fitness)
    # pool.close()
    # pool.join()


    # print(type(fitness))
    bestfitness = np.min(fitness)
    bestindex = np.where(fitness == np.min(fitness))[0][0]
    # [bestfitness,bestindex]=min(fitness)
    zbest = copy.deepcopy(pop[bestindex,:])
    gbest = pop
    fitnessgbest = fitness
    fitnesszbest = bestfitness
    # print(pop,bestfitness, bestindex,zbest,gbest,fitnessgbest,fitnesszbest)
    # print('-----------------------------------------')

    for i in range(maxg):
    # ------------------------------------------------------------------------------
        for j in range(sizepop):

            V[j,:] = V[j,:] + c1 * np.random.random() * (gbest[j,:] - pop[j,:]) + c2 * np.random.random() * (zbest - pop[j,:])
            # np.minimum(V,Vmax)
            # np.maximum(V, Vmax)
            pop[j,:]=pop[j,:]+0.001* V[j,:]
            if (pop[j, 0] > popmax[0]): pop[j, 0] =popmax[0]
            if (pop[j, 1] > popmax[1]): pop[j, 1] =popmax[1]
            if (pop[j, 0] < popmin[0]): pop[j, 0] =popmin[0]
            if (pop[j, 1] < popmin[1]): pop[j, 1] =popmin[1]

            # 自适应变异
            if(np.random.random()>0.8):
                k = int(np.random.random()*paranum)
                pop[j, k] = np.random.random()*(popmax[k]-popmin[k])+popmin[k]
            #  适应度值
            # fitness[j] = fun(pop[j,:])
        # fitness[j] = fun(pop[j, :])
        with multiprocessing.Pool(processes=cpu_num) as pool:
            fitness = pool.map(fun, pop)
        print(fitness)
        # pool = multiprocessing.Pool(processes=cpu_num)
        # fitness=pool.map(fun,pop)
        # pool.close()
        # pool.join()
        # print(fitness)

        for j in range(sizepop):
            if fitness[j] < fitnessgbest[j]:
                gbest[j, :] = pop[j, :]
                fitnessgbest[j] = fitness[j]
            if fitness[j] < fitnesszbest:
                zbest = copy.deepcopy(pop[j, :])
                fitnesszbest = fitness[j]
    # ------------------------------------------------------------------------------
        yy[i] = fitnesszbest
        # if i%20==0:
        #     print(i)
        #     try:
        #         ax.lines.remove(lines[0])
        #     except Exception:
        #         pass
            # plot the prediction
            # y, t, step1 = fun(zbest)
            # lines = ax.plot(lam[0:len(lam):step1], t[0:len(lam):step1] , 'r-', lw=2)
            # plt.pause(0.1)
    # plt.ioff()
    toc = time.time()

    print('最优值：', zbest)

    print('MF:', fitnesszbest)
    print(fun(zbest))
    print('time:', toc-tic)
    # print(yy)
    # print(bestindex)
    fig=plt.figure()
    plt.plot(range(maxg),yy)
    plt.show()






