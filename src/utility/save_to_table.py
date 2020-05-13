import codecs


def save_values_to_table(results, sizes, fn, caption):
    with codecs.open(f'tbl/{fn}.tex', 'w', 'utf-8') as out:
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
        out.write('\t\\caption{' + caption + '}\n')
        out.write('\\end{table}\n')


def save_intervals_to_table(results, sizes, fn, caption):
    with codecs.open(f'tbl/{fn}.tex', 'w', 'utf-8') as out:
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
        out.write('\t\\caption{' + caption + '}\n')
        out.write('\\end{table}\n')
