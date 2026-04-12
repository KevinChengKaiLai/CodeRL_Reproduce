
cd /workspace/CodeRL
mkdir -p data/APPS
cd data/APPS

# Download (~1.3GB tar.gz)
wget https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz

# Extract
tar -xzf APPS.tar.gz

# This gives you:
# data/APPS/train/0000/  data/APPS/train/0001/ ... (5000 problems)
# data/APPS/test/0000/   data/APPS/test/0001/  ... (5000 problems)

# Each problem folder contains:
# question.txt       ← problem description
# solutions.json     ← list of correct Python solutions
# input_output.json  ← unit test cases
# metadata.json      ← difficulty level

# Clean up the archive
rm APPS.tar.gz