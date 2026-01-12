from tree_sitter import Language
import os

os.makedirs("build", exist_ok=True)

# 直接修改这里的后缀
output_file = 'build/my-languages-32.dll'

Language.build_library(
    output_file,
    [
        'F:/CPDP/tree-sitter-c'
    ]
)

print(f"✅ Tree-sitter C parser compiled successfully to {output_file}")