from matplotlib import pyplot as plt


def save_values_to_image(results, sizes, fn, styles, title):
    plt.figure(figsize=(10, 5))
    plt.xscale('log')
    plt.yscale('log')
    for style, (algo, result) in zip(styles, results.items()):
        plt.plot(sizes, result.values(), style, label=algo)
    plt.xlabel('Розмір задачі', fontsize=16)
    plt.ylabel(title, fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(which='both')
    plt.savefig(f'img/{fn}.png', bbox_inches='tight')
    plt.show()


def save_intervals_to_image(results, sizes, fn, styles, title):
    plt.figure(figsize=(10, 5))
    plt.xscale('log')
    plt.yscale('log')
    for style, (algo, result) in zip(styles, results.items()):
        plt.plot(sizes, [r[0] for r in result.values()], style, label=algo)
        plt.fill_between(sizes,
                         [r[0] - r[1] / 5 for r in result.values()], 
                         [r[0] + r[1] / 5 for r in result.values()],
                         color=style[0], alpha=.1)
    plt.xlabel('Розмір задачі', fontsize=16)
    plt.ylabel(title, fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(which='both')
    plt.savefig(f'img/{fn}.png', bbox_inches='tight')
    plt.show()
