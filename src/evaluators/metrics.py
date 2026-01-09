# metrics.py

class MetricsCalculator:
    def __init__(self):
        # 定义哪些语法节点代表“分支/决策点”
        # 这是 C 语言常见的控制流节点
        self.branch_nodes = {
            'if_statement',
            'while_statement',
            'for_statement',
            'case_statement',
            'do_statement'
            # 注意：严谨的计算可能还包括 && 和 || 运算符，这里先做基础版
        }

    def compute_cyclomatic_complexity(self, node):
        """
        计算圈复杂度 (Cyclomatic Complexity)
        公式: 复杂度 = 判定节点数量 + 1
        """
        complexity = 1

        # 内部递归函数来遍历子节点
        def traverse(n):
            nonlocal complexity
            if n.type in self.branch_nodes:
                complexity += 1

            # 继续遍历子节点
            for child in n.children:
                traverse(child)

        traverse(node)
        return complexity

    def count_parameters(self, function_node):
        """
        计算函数参数个数
        """
        # 在 function_definition 下寻找 parameter_list
        # 使用简单的遍历查找
        param_count = 0

        # 1. 找到 declarator (它可能包裹在多层结构中，视具体语法而定)
        # 简单策略：在函数节点内直接用 query 找 parameter_list 的直接子节点

        # 为了演示简单逻辑，我们遍历一下找 parameter_list
        param_list_node = None

        # 广度优先或深度搜索找到 parameter_list
        # 这是一个简化版，通常 parameter_list 在 function_declarator 下
        cursor = function_node.walk()

        # 这里为了演示简单性，我们假设传入的是 function_definition
        # 我们直接搜集所有的 parameter_declaration
        for child in function_node.children:
            # 这里需要根据具体 AST 结构深入，为简化我们暂时只返回 0
            # 实际项目中建议使用 Query 来做，下面 Evaluation.py 会演示 Query 方法
            pass

        return 0  # 占位，建议在外部用 Query 实现

    def get_loc(self, node):
        """
        计算代码行数 (Lines of Code)
        """
        return node.end_point[0] - node.start_point[0] + 1