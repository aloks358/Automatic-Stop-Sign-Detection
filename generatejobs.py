import os
import sys

TEMPLATE_SH = 'example.sh'
IMAGES_PATH = 'IMAGES'
SCRIPTS_PATH = 'QSUB_SCRIPTS'
SCRIPTS_PREFIX = 'subm'

FULL_PREFIX = os.path.join(SCRIPTS_PATH, SCRIPTS_PREFIX)

with open(TEMPLATE_SH, 'r') as f:
    template_str = f.read()

# Last character is a new line that needs to be removed
template_str = template_str[:-1]

i = 0
paths = os.listdir(IMAGES_PATH)
for path in paths:
    path = os.path.join(IMAGES_PATH, path)
    print i
    with open(FULL_PREFIX + str(i) + '.sh', 'w') as f:
        f.write(template_str + ' ' + path + '\n')
    os.system('qsub ' + FULL_PREFIX + str(i) + '.sh')
    i = i + 1
