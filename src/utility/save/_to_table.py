import codecs


def save_values_to_table(results, sizes, fp, tp):
    with codecs.open(f'tbl/{fp}/{tp}.tex', 'w', 'utf-8') as out:
        # head
        out.write('\\begin{table}[H]\n')
        out.write('\t\\centering\n')

        # body
        out.write('\t\\begin{tabular}{|c||' + 'c|' * len(sizes) + '}' + '\\hline\n')
        out.write('\t\tРозмір задачі & ' + ' & '.join(map(str, sizes)) + ' \\\\ \\hline \\hline\n')
        for algo, result in results.items():
            out.write('\t\t' + algo + ' & ' +
                      ' & '.join(map(lambda r: f'{r:.2f}'
                                     if type(r) is float else str(r), result.values())) +
                      ' \\\\ \\hline\n')
        out.write('\t\\end{tabular}\n')

        # tail
        if tp == 'time':
            out.write('\t\\caption{Час виконання, секунд}\n')
        if tp == 'iter':
            out.write('\t\\caption{Число ітерацій}\n')
        out.write('\\end{table}\n')


def save_intervals_to_table(results, sizes, fn, tp):
    with codecs.open(f'tbl/{fp}/{tp}.tex', 'w', 'utf-8') as out:
        # head
        out.write('\\begin{table}[H]\n')
        out.write('\t\\centering\n')

        # body
        end = ' \\\\ \\hline\n'
        out.write('\t\\begin{tabular}{|c||' + 'c|' * len(sizes) + '}' + '\\hline\n')
        out.write('\t\tРозмір задачі & ' + ' & '.join(map(str, sizes)) + ' \\\\ \\hline \\hline\n')
        for algo, result in results.items():
            out.write('\t\t' + algo + ' & ' +
                      ' & '.join(map(lambda r: f'{r[0]:.2f} $\\pm$ {r[1]:.2f}', result.values())) +
                      ' \\\\ \\hline\n')
        out.write('\t\\end{tabular}\n')

        # tail
        if tp == 'time':
            out.write('\t\\caption{Час виконання, секунд}\n')
        if tp == 'iter':
            out.write('\t\\caption{Число ітерацій}\n')
        out.write('\\end{table}\n')
