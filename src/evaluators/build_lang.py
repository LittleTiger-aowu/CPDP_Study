
from tree_sitter import Language
import os

os.makedirs("build", exist_ok=True)

# 修改 1: 输出文件名最好改一下，避免和 C 语言的混淆
# 建议叫 java_lang.dll
output_file = ('build/java_lang.dll')

Language.build_library(
    output_file,
    [
        # 修改 2: 指向刚才下载的 Java 语法文件夹
        # 必须确保这个文件夹里有 src/parser.c 和 grammar.js
        'F:/CPDP/tree-sitter-java'
    ]
)

print(f"✅ Tree-sitter JAVA parser compiled successfully to {output_file}")