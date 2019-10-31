import matplotlib.pyplot as plt
import numpy as np


def plot_isc(isc_all):
    # plot ISC as a bar chart
    plt.figure()
    comp1 = [cond['ISC'][0] for cond in isc_all.values()]
    comp2 = [cond['ISC'][1] for cond in isc_all.values()]
    comp3 = [cond['ISC'][2] for cond in isc_all.values()]
    barWidth = 0.2
    r1 = np.arange(len(comp1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    plt.bar(r1, comp1, color='gray', width=barWidth, edgecolor='white', label='Comp1')
    plt.bar(r2, comp2, color='green', width=barWidth, edgecolor='white', label='Comp2')
    plt.bar(r3, comp3, color='green', width=barWidth, edgecolor='white', label='Comp3')
    plt.xticks([r + barWidth for r in range(len(comp1))], isc_all.keys())
    plt.ylabel('ISC', fontweight='bold')
    plt.title('ISC for each condition')
    plt.legend()
    plt.show()

    # plot ISC_persecond
    plt.figure()
    for cond in isc_all.values():
        plt.plot(cond['ISC_persecond'][0])
        plt.legend(isc_all.keys())
        plt.xlabel('Time (s)')
        plt.ylabel('ISC')
        plt.title('ISC per second for each condition')


    # plot ISC_bysubject
    fig, ax = plt.subplots()
    ax.set_title('ISC by subject for each condition')
    a = [cond['ISC_bysubject'][0, :] for cond in isc_all.values()]
    ax.set_xticklabels(isc_all.keys())
    ax.set_ylabel('ISC')
    ax.set_xlabel('Conditions', fontweight='bold')
    ax.boxplot(a)

plot_isc(isc_results)

