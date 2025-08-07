import os

# Create directories if they don't exist
def mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)
        print(f"Created directory: {p}")

# Create symbolic links from source to destination
def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst, target_is_directory=True)
        print(f"Linked {src} -> {dst}")
    else:
        print(f"Link already exists: {dst}")

# Define directories
base_path = '/'
train_path_from = os.path.abspath(os.path.join(base_path, 'fruits-360-100x100/Training'))
test_path_from = os.path.abspath(os.path.join(base_path, 'fruits-360-100x100/Test'))

# Define destination paths for reduced dataset
train_path_to = os.path.abspath(os.path.join(base_path, 'fruits-360-100x100-small/Training'))
test_path_to = os.path.abspath(os.path.join(base_path, 'fruits-360-100x100-small/Test'))

# Create destination directories if they don't exist
mkdir(train_path_to)
mkdir(test_path_to)

# Define the subset of classes
classes = [
    'Apple Golden 1',
    'Avocado 1',
    'Lemon 1',
    'Mango 1',
    'Kiwi 1',
    'Banana 1',
    'Strawberry 1',
    'Raspberry 1'
]

# Process each class
for c in classes:
    # Source and destination for Training data
    src_train = os.path.join(train_path_from, c)
    dst_train = os.path.join(train_path_to, c)

    # Source and destination for test data
    src_test = os.path.join(test_path_from, c)
    dst_test = os.path.join(test_path_to, c)

    # Check if the source directories exist before linking
    if os.path.exists(src_train):
        link(src_train, dst_train)
    else:
        print(f"Training directory does not exist for class: {c} ({src_train})")

    if os.path.exists(src_test):
        link(src_test, dst_test)
    else:
        print(f"Test directory does not exist for class: {c} ({src_test})")

# Print the final contents of the destination directories
print("\nFinal content of Training folder:")
print(os.listdir(train_path_to))

print("\nFinal content of Validation folder:")
print(os.listdir(test_path_to))
