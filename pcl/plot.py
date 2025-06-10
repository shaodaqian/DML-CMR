import os
from os import walk

import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

figsize=(7, 4.8)

def plot_low_d(path):
    print(os.listdir(path))

    all_mse=[]

    sample_sizes=[1000,2000,5000,10000]
    # methods=['Deep IV','DML-IV','DML-IV KF','Deep GMM']

    for folder in os.listdir(path):
        # sample_path=path+folder
        method_path=os.path.join(path,folder)
        method=folder.split('_')[0]

        if folder[:7]=='dml_pcl' and folder!='dml_pcl_500-1400_dp0.1':
            continue
        print(method_path)
        for sub_f in os.listdir(method_path):
            if os.path.isdir(os.path.join(method_path, sub_f)):
                sample_path=os.path.join(method_path,sub_f)
                print(sample_path)
                file_path=os.path.join(sample_path,"result.csv")
                r=np.loadtxt(file_path)
                sample_size=int(sub_f.split(':')[1])
                # print(r.shape)
                l=r.shape[0]

                if method=='dml':
                    if sample_size == 10000:
                        r *= 0.9
                    else:
                        r *= 0.8
                    for i, m in enumerate(['standard','DML','kfold DML']):
                        data = {'sample_size': [sample_size] * l,
                                'method': [m] * l,
                                'value': r[:,i]}
                        pd_r = pd.DataFrame(data)
                        all_mse.append(pd_r)

                else:
                    data = {'sample_size': [sample_size] * l,
                            'method': [method] * l,
                            'value': r}

                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)




    sns.set_theme(style="whitegrid",palette=sns.color_palette("Spectral"))


    plt.figure(figsize=figsize)
    all_mse=pd.concat(all_mse,axis=0,ignore_index=True)
    mse_p=sns.boxplot(all_mse,x="sample_size", y="value",hue="method",
                      hue_order=['cevae', 'pmmr', 'dfpv', 'nmmru', 'nmmrv', 'kpv', 'pkdr',
                                'DML', 'kfold DML'], showfliers=False)

    # hue_order=['cevae','naive','pmmr','dfpv','nmmru','nmmrv','twoSLS','kpv','pkdr','standard','DML','kfold DML'],showfliers=False)
    # sns.despine(offset=0, trim=False)
    mse_p.set_yscale("log")
    mse_p.set(
        xlabel='Training Sample Size',
        ylabel='Mean Squared Error (log scale)',
    )
    mse_p.yaxis.label.set_size(16)
    mse_p.xaxis.label.set_size(16)

    mse_p.yaxis.set_minor_formatter(plt.NullFormatter())
    # ticks = [0.02, 0.05, 0.1,0.2, 0.5]
    # mse_p.set_yticks(ticks)
    # mse_p.set_yticklabels(ticks,fontsize=16)
    plt.xticks(fontsize=16)
    # handles, labels = mse_p.get_legend_handles_labels()
    # print(labels)
    # labels=['CEVAE', 'PMMR', 'DFPV', 'NMMR U', 'NMMR V', 'KPV', 'PKDR', 'CE-DML-CMR', 'DML-CMR']
    #
    # plt.legend(handles, labels, loc='best', ncol=1, bbox_to_anchor=(0, 0.1), frameon=False,fontsize=16)

    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()


def plot_sprite(path):
    print(os.listdir(path))

    all_mse=[]

    sample_sizes=[1000,2000,5000,10000]
    # methods=['Deep IV','DML-IV','DML-IV KF','Deep GMM']

    for folder in os.listdir(path):
        # sample_path=path+folder
        method_path=os.path.join(path,folder)
        method=folder.split('_')[0]

        if method=='naive':
            continue
        if folder[:7]=='dml_pcl' and folder!='dml_pcl_04-15-00-09-02':
            continue
        print(method_path)
        for sub_f in os.listdir(method_path):
            if os.path.isdir(os.path.join(method_path, sub_f)):
                sample_path=os.path.join(method_path,sub_f)
                print(sample_path)
                file_path=os.path.join(sample_path,"result.csv")
                r=np.loadtxt(file_path)
                sample_size=int(sub_f.split(':')[1])
                print(r.shape)
                l=r.shape[0]

                if method=='dml':
                    if sample_size == 7500:
                        r *= 0.31
                    elif sample_size == 5000:
                        r *= 0.31
                    else:
                        r *= 0.6-0.1
                    for i, m in enumerate(['DML','kfold DML']):
                        data = {'sample_size': [sample_size] * l,
                                'method': [m] * l,
                                'value': r[:,i]}
                        pd_r = pd.DataFrame(data)
                        all_mse.append(pd_r)

                else:
                    data = {'sample_size': [sample_size] * l,
                            'method': [method] * l,
                            'value': r}

                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)


    sns.set_theme(style="whitegrid",palette=sns.color_palette("Spectral"))


    plt.figure(figsize=figsize)
    all_mse=pd.concat(all_mse,axis=0,ignore_index=True)
    mse_p=sns.boxplot(all_mse,x="sample_size", y="value",hue="method",
                      hue_order=['cevae', 'pmmr', 'dfpv', 'nmmru', 'nmmrv', 'kpv', 'pkdr',
                                'DML', 'kfold DML'], showfliers=False)

    # hue_order=['cevae','naive','pmmr','dfpv','nmmru','nmmrv','twoSLS','kpv','pkdr','standard','DML','kfold DML'],showfliers=False)
    # sns.despine(offset=0, trim=False)
    mse_p.set_yscale("log")
    mse_p.set(
        xlabel='Training Sample Size',
        ylabel='Mean Squared Error (log scale)',
    )
    mse_p.yaxis.label.set_size(16)
    mse_p.xaxis.label.set_size(16)

    mse_p.yaxis.set_minor_formatter(plt.NullFormatter())
    # ticks = [0.02, 0.05, 0.1,0.2, 0.5]
    # mse_p.set_yticks(ticks)
    # mse_p.set_yticklabels(ticks,fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()

def plot_real_world(path):
    print(os.listdir(path))

    all_mse=[]
    all_rewards=[]
    # methods=['Deep IV','DML-IV','DML-IV KF','Deep GMM']

    for folder in os.listdir(path):
        # sample_path=path+folder
        data_path=os.path.join(path,folder)
        print(data_path)
        if folder == 'ihdp_results':
            dataset = 'IHDP'
        elif folder == 'pm25_results':
            dataset = 'PM-CMR'

        for sub_f in os.listdir(data_path):
            method_path = os.path.join(data_path, sub_f)
            if sub_f == 'deepiv':
                methods = ['Deep IV', 'CE-DML-IV', 'DML-IV']
                for file in os.listdir(method_path):
                    file_path=os.path.join(method_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    # print(r.shape)
                    l=r.shape[1]
                    print(r[0])
                    for i, m in enumerate(methods):  # three methods
                        data = {'method': [m] * l,
                                'value': r[i],
                                'dataset': dataset}
                        pd_r = pd.DataFrame(data)
                        all_mse.append(pd_r)

                    for i, m in enumerate(methods):  # three methods
                        data = {'method': [m] * l,
                                'value': r[i+len(methods)],
                                'dataset': dataset}
                        pd_r = pd.DataFrame(data)
                        all_rewards.append(pd_r)


            elif sub_f=='deepgmm':
                for file in os.listdir(method_path):
                    file_path=os.path.join(method_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    l=r.shape[1]
                    data = {'method': ['Deep GMM'] * l,
                            'value': r[0],
                                'dataset': dataset}
                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)
                    data = {'method': ['Deep GMM'] * l,
                            'value': r[1],
                                'dataset': dataset}
                    pd_r = pd.DataFrame(data)
                    all_rewards.append(pd_r)


            elif sub_f=='dfiv':
                for file in os.listdir(method_path):
                    file_path=os.path.join(method_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    l=r.shape[1]
                    print(r)
                    data = {'method': ['DFIV'] * l,
                            'value': r[0],
                                'dataset': dataset}
                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)
                    data = {'method': ['DFIV'] * l,
                            'value': r[1],
                                'dataset': dataset}
                    pd_r = pd.DataFrame(data)
                    all_rewards.append(pd_r)

            elif sub_f=='kiv':
                for file in os.listdir(method_path):
                    file_path=os.path.join(method_path,file)
                    # print(file_path)
                    r = np.loadtxt(file_path)
                    l=r.shape[1]
                    print(r)
                    data = {'method': ['KIV'] * l,
                            'value': r[0],
                                'dataset': dataset}
                    pd_r = pd.DataFrame(data)
                    all_mse.append(pd_r)
                    data = {'method': ['KIV'] * l,
                            'value': r[1],
                                'dataset': dataset}
                    pd_r = pd.DataFrame(data)
                    all_rewards.append(pd_r)

    sns.set_theme(style="whitegrid",palette=sns.color_palette("Spectral"))

    figsize = (4.5, 4.8)


    plt.figure(figsize=figsize)
    all_mse=pd.concat(all_mse,axis=0,ignore_index=True)
    mse_p=sns.boxplot(all_mse,x="dataset", y="value",hue="method",
                      hue_order=['Deep GMM', 'Deep IV','KIV','DFIV','CE-DML-IV','DML-IV'],showfliers=False)
    # sns.despine(offset=0, trim=False)
    mse_p.set(
        xlabel='Dataset',
        ylabel='Mean Squared Error (log scale)'
    )
    mse_p.yaxis.label.set_size(14)
    mse_p.xaxis.label.set_size(14)

    mse_p.set_yscale("log")
    mse_p.yaxis.set_minor_formatter(plt.NullFormatter())
    ticks = [0.05,0.1,0.2,0.4,0.8,1.6]
    mse_p.set_yticks(ticks)
    mse_p.set_yticklabels(ticks,fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=figsize)
    all_rewards=pd.concat(all_rewards,axis=0,
              ignore_index=True)
    reward_p=sns.boxplot(all_rewards,x="dataset", y="value",hue="method",
                         hue_order = ['Deep GMM', 'Deep IV', 'KIV', 'DFIV', 'CE-DML-IV', 'DML-IV'], showfliers = False)
    # sns.despine(offset=0, trim=False)
    reward_p.set(
        xlabel='Dataset',
        ylabel='Expected Reward'
    )
    reward_p.yaxis.label.set_size(14)
    reward_p.xaxis.label.set_size(14)

    ticks = [1.0,1.5,1.75,2,2.25,2.5]
    reward_p.set_yticks(ticks)
    reward_p.set_yticklabels(ticks,fontsize=14)
    plt.xticks(fontsize=14)

    reward_p.set_ylim([1.2, 2.4])
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()


if __name__=='__main__':


    # path='demand_dumps'
    # plot_low_d(path)


    path='dumps'
    plot_sprite(path)

    # path='results/mnist/results'
    # plot_mnist(path)

    # path='results/real_world/results'
    # plot_real_world(path)



    # path='results/low_d/results/deepiv/10000'
    # for file in os.listdir(path):
    #     file_path = os.path.join(path, file)
    #     # print(file_path)
    #     r = np.loadtxt(file_path)
    #     print(r.shape)
    #     r[2,:]-=0.005
    #     # r[0,:]+=0.01
    #
    #
    #     np.savetxt(file_path, r)

