#!/bin/bash
for f in checkpoint.py dataset.py env_setup.py model.py sample.py utils.py cli.py diffusion.py graph_utils.py train.py; do
  echo "--- $f ---"
  python -c "import ast; ast.parse(open('$f').read())" && echo "OK"
done
