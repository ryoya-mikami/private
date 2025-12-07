
import json
import numpy as np

nb_path = r"c:\Users\ryoya\OneDrive\Documents\課題\プログラミング\private\free\bonoroizu.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'vor.points.ptp().max()' in line:
                new_line = line.replace('vor.points.ptp().max()', 'np.ptp(vor.points).max()')
                new_source.append(new_line)
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Fixed bonoroizu.ipynb")
