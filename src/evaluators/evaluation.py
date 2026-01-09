# #
# # from tree_sitter import Language, Parser
# # import os
# #
# # # --- 1. 配置路径与语言 ---
# # # 注意：这里后缀改成了 .dll
# # LIB_PATH = os.path.join("build", "my-languages.dll")
# #
# # # 注意：你编译的是 tree-sitter-c，所以语言内部名称通常是 'c'
# # LANGUAGE_NAME = 'c'
# #
# # def main():
# #     # --- 2. 加载语言库 ---
# #     if not os.path.exists(LIB_PATH):
# #         print(f"❌ 错误：找不到文件 {LIB_PATH}")
# #         return
# #
# #     try:
# #         # 加载我们刚编译好的 .dll
# #         c_lang = Language(LIB_PATH, LANGUAGE_NAME)
# #         print("✅ 语言库加载成功！")
# #     except Exception as e:
# #         print(f"❌ 加载失败: {e}")
# #         return
# #
# #     # --- 3. 初始化解析器 ---
# #     parser = Parser()
# #     parser.set_language(c_lang)
# #
# #     # --- 4. 准备一段 C 语言测试代码 ---
# #     # 注意：Tree-sitter 只接受 bytes 类型，所以前面加了 b
# #     source_code = b"""
# #     #include <stdio.h>
# #
# #     int main() {
# #         int a = 10;
# #         printf("Hello Tree-sitter! Number: %d", a);
# #         return 0;
# #     }
# #     """
# #
# #     # --- 5. 解析并生成 AST ---
# #     tree = parser.parse(source_code)
# #     root_node = tree.root_node
# #
# #     # --- 6. 验证结果 ---
# #     print("\n--- AST 结构 (S-expression) ---")
# #     # sexp() 会以字符串形式打印树结构，能看到这就是 C 语言的结构
# #     print(root_node.sexp())
# #
# #     print("\n--- 简单遍历根节点的子节点 ---")
# #     for child in root_node.children:
# #         # 提取节点对应的源代码文本
# #         code_text = source_code[child.start_byte : child.end_byte].decode('utf-8')
# #         print(f"类型: {child.type:<20} | 内容: {code_text}")
# #
# # if __name__ == "__main__":
# #     main()
# from tree_sitter import Language, Parser
# import os
#
# # --- 配置 ---
# LIB_PATH = os.path.join("build", "my-languages.dll")
# LANGUAGE_NAME = 'c'
#
#
# def main():
#     # 1. 初始化
#     c_lang = Language(LIB_PATH, LANGUAGE_NAME)
#     parser = Parser()
#     parser.set_language(c_lang)
#
#     # 2. 稍微复杂一点的 C 代码 (包含两个函数)
#     source_code = b"""
#     #include <stdio.h>
#
#     int add(int a, int b) {
#         return a + b;
#     }
#
#     int main() {
#         int result = add(5, 10);
#         printf("Result: %d", result);
#         return 0;
#     }
#     """
#
#     tree = parser.parse(source_code)
#
#     # --- 3. 使用 Query (查询) ---
#     # 这段类似 Lisp 的语法就是 Query。
#     # 它的意思是：找到所有的 function_definition，
#     # 并且把其中的 function_declarator -> identifier 标记为 @func_name
#
#     query_scm = """
#     (function_definition
#       declarator: (function_declarator
#         declarator: (identifier) @func_name
#       )
#     )
#     """
#
#     query = c_lang.query(query_scm)
#
#     # 4. 执行查询
#     captures = query.captures(tree.root_node)
#
#     print(f"我们在代码中发现了 {len(captures)} 个函数定义：")
#     print("-" * 30)
#
#     for node, tag_name in captures:
#         if tag_name == 'func_name':
#             # 获取函数名文本
#             func_name_text = source_code[node.start_byte: node.end_byte].decode('utf-8')
#
#             # 获取函数所在的行号 (start_point 是 (行, 列) 元组)
#             line_number = node.start_point[0] + 1
#
#             print(f"函数名: {func_name_text:<10} | 位于第 {line_number} 行")
#
#
# if __name__ == "__main__":
#     main()

# evaluation.py
from tree_sitter import Language, Parser
import os
from metrics import MetricsCalculator  # 导入刚才写的类

# --- 配置 ---
LIB_PATH = os.path.join("build", "my-languages.dll")  # 确保是 .dll
LANGUAGE_NAME = 'c'


def main():
    # 1. 准备解析器
    if not os.path.exists(LIB_PATH):
        print(f"找不到库文件: {LIB_PATH}")
        return

    c_lang = Language(LIB_PATH, LANGUAGE_NAME)
    parser = Parser()
    parser.set_language(c_lang)

    # 初始化度量计算器
    calculator = MetricsCalculator()

    # 2. 测试代码 (包含不同复杂度的函数)
    source_code_str = """#include <stdio.h>

// 简单的函数: 复杂度 1
int add(int a, int b) {
    return a + b;
}

// 复杂的函数: 包含 if, for, while
void complex_logic(int n) {
    int i = 0;
    if (n > 0) {            // +1
        for (i=0; i<n; i++) { // +1
            if (i % 2 == 0) { // +1
                printf("Even");
            }
        }
    } else {
        while (n < 0) {     // +1
            n++;
        }
    }
}"""
    source_code = source_code_str.encode('utf-8')

    tree = parser.parse(source_code)

    # 3. 使用 Query 提取所有"函数定义"
    query_scm = """
    (function_definition
      declarator: (function_declarator
        declarator: (identifier) @func_name
      )
    ) @func_body
    """
    query = c_lang.query(query_scm)
    captures = query.captures(tree.root_node)

    print(f"{'函数名':<15} | {'LOC':<5} | {'圈复杂度':<10}")
    print("-" * 35)

    # captures 返回的是一个 list，包含所有匹配到的节点
    # 我们需要过滤出 @func_name 和 @func_body
    # Tree-sitter 的 captures 格式是 [(Node, "tag_name"), ...]

    # 这里的逻辑稍显复杂，因为 captures 是扁平的列表
    # 为了方便，我们只遍历 @func_body (整个函数节点)

    # 重新定义 Query 只抓取整个函数体，然后在循环内部找名字
    # 简化版逻辑：

    processed_nodes = set()

    for node, tag_name in captures:
        if tag_name == 'func_body':
            # 去重：防止同一个节点被处理多次
            if node.id in processed_nodes:
                continue
            processed_nodes.add(node.id)

            # --- A. 获取函数名 ---
            # 在当前函数节点下找名字有点麻烦，通常我们在 Query 里通过 @func_name 捕获
            # 这里为了演示，我们简单截取名字（实际项目应通过对应的 @func_name 节点获取）
            # 下面是一个稍微 hack 的方法来通过 captures 列表配对，或者直接解析文本

            # 为了代码简洁，我们假设我们能找到名字
            func_name = "Unknown"
            # 查找该节点的子节点里的 identifier (这只是简易演示)
            # 正确做法是配对处理 captures 列表

            # --- B. 计算指标 ---
            loc = calculator.get_loc(node)
            complexity = calculator.compute_cyclomatic_complexity(node)

            print(f"{'Function':<15} | {loc:<5} | {complexity:<10}")

    # --- 优化：更精准的名称获取演示 ---
    # 如果你想精准匹配名字和函数体，可以遍历 capture 列表并自行组装
    # 但为了让你先跑通 metrics，你可以先关注上面的复杂度数值是否正确。


if __name__ == "__main__":
    main()