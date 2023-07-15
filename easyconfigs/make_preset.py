import os

files = os.listdir('.')

presets = {}

for file in files:
    if file.endswith('yaml'):
        with open(file) as f:
            presets[file[:-5]] = f.read()

with open('../mrfi/easyconfig_presets.py', 'w') as f:
    print('easyconfig_presets = {', file = f)

    for name, yamlcfg in presets.items():
        print(f'"{name}":"""\n{yamlcfg}""",', file = f)

    print('}', file = f)
