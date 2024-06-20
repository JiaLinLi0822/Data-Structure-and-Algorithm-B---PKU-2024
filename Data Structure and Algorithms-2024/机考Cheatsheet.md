## 0. 常用标准库

```python
### 查看提示信息
>> help(func)

### 快速读写
import sys
input = lambda: sys.stdin.readline().strip()
write = lambda x: sys.stdout.write(str(x))

### 格式转换
print('{:.2f}'.format(num))

### 字符串匹配
iterable.count(value)
str.find(sub)		#未找到抛出-1
list.index(x)		#未找到抛出ValueError

### 多组数据读入
try

except EOFerror

### math
math.pow(x, y) == x**y
math.factorial(n) == n!

### 排列组合
from itertools import permutations,combinations
permutations(list)			#生成list的全排列（每个以元组形式存在）
combinations(list,k)		#生成list的k元组合（无序）（每个以元组形式存在）

### 优先队列
from queue import PriorityQueue
q = PriorityQueue()					#创建PriorityQueue对象
q.put((priority number, data))		#存入数据，其中priority number越小代表优先级越大
q.empty()							#判断优先队列是否为空
q.get()								#弹出优先级最高的优先级和元素（以元组的形式）
q.qsize()							#返回优先队列的大小

### 默认值字典
from collections import defaultdict
dic = defaultdict(key_type)				#初始化时须指定值的类型
dic = defaultdict(lambda: default_value)		#不使用默认初始值

### 计数字典
from collections import Counter
c = Counter(list)			#返回计数字典

### 二分查找
import bisect
bisect.bisect_left(lst,x)
# 使用bisect_left查找插入点，若x∈lst，返回最左侧x的索引；否则返回最左侧的使x若插入后能位于其左侧的元素的当前索引。
bisect.bisect_right(lst,x)
# 使用bisect_right查找插入点，若x∈lst，返回最右侧x的索引；否则返回最右侧的使x若插入后能位于其右侧的元素的当前索引。
bisect.insort(lst,x)
# 使用insort插入元素，返回插入后的lst
```

## 1. 栈

### 1.1 波兰表达式/逆波兰表达式

1. 从右到左扫描表达式（逆波兰从左到右扫描）。
2. 遇到操作数时，将其压入堆栈。
3. 遇到运算符时，从堆栈中弹出相应数量的操作数，进行计算，并将结果压入堆栈。
4. 扫描完整个表达式后，堆栈顶端的值就是表达式的结果。

```python
a = list(input().split())
f = ['+', '-', '*', '/']
s =[]
for i in range(len(a)):
    if a[i] not in f:
        a[i] = float(a[i])

for i in range(len(a), 0, -1):
    if a[i-1] in f:
        if a[i-1] == '+':
            s.append(s.pop() + s.pop())
        elif a[i-1] == '-':
            s.append(s.pop() - s.pop())
        elif a[i-1] == '*':
            s.append(s.pop() * s.pop())
        elif a[i-1] == '/':
            s.append(s.pop() / s.pop())
    else:
        s.append(a[i-1])
print("{:.6f}".format(float(s.pop()))) #保留六位小数
```

### 1.2 中序转后序表达式(Shunting Yard算法)

1. 初始化两个空栈：一个**操作符栈**和一个**输出栈**。
2. 从**左到右**扫描中缀表达式：
   - 如果遇到操作数，将其直接放入输出栈。
   - 如果遇到操作符，判断其优先级：
     - 如果操作符栈为空，或栈顶操作符的优先级低于当前操作符，将当前操作符压入操作符栈。
     - 否则，将操作符栈顶的操作符弹出并放入输出栈，重复该步骤直到当前操作符可以压入操作符栈。
   - 如果遇到**左括号**，将其压入操作符栈。
   - 如果遇到右括号，一直弹出操作符栈顶的操作符并放入输出栈，直到遇到左括号，将左括号弹出但不放入输出栈。
3. 如果扫描完表达式后操作符栈中还有操作符，将其依次弹出并放入输出栈。

```python
def infix_to_postfix(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '(': 0} #优先级：乘除大于加减
    output = [] #输出栈
    operators = [] #操作符栈

    tokens = expression.split()

    for token in tokens:
        if token.isnumeric(): #判断字符是否为数字
            output.append(token) #操作数直接放入输出栈
        elif token == '(': #左括号压入操作符栈
            operators.append(token)
        elif token == ')': #右括号
            top_token = operators.pop() #弹出顶部操作符
            while top_token != '(': #只要不是左括号就弹出并放入输出栈，左括号只弹出
                output.append(top_token) 
                top_token = operators.pop()
        else: #遇到加减乘除操作符，判断优先级
          #操作符栈不为空且栈顶操作符优先级高于当前操作符，将操作符栈中的操作符弹出直至满足要求
            while (operators and precedence[operators[-1]] >= precedence[token]): 
                output.append(operators.pop())
            operators.append(token) #操作符栈为空或栈顶优先级低于当前操作符，直接压入栈
#扫描完还有操作符栈还有操作符，依次弹出
    while operators:
        output.append(operators.pop())

    return ' '.join(output)
```

### 1.3 单调栈

**找到数组中每个元素右侧第一个比它大的元素（或下标）。**

1. 从左到右遍历数组。
2. 对于每个元素，如果栈不为空且当前元素大于栈顶元素，则弹出栈顶元素，并将当前元素作为弹出元素的右侧第一个比它大的元素。
3. 将当前元素入栈。
4. 遍历结束后，栈中剩余的元素在其右侧没有比它们大的元素。

```python
def next_greater_elements(nums):
    stack = []
    result = [-1] * len(nums)  # 初始化结果数组

    for i, num in enumerate(nums):
        while stack and nums[stack[-1]] < num:
            index = stack.pop()
            result[index] = num
        stack.append(i)
    
    return result
  ### The next greater elements of [2, 1, 2, 4, 3] are: [4, 2, 4, -1, -1]
```

给出项数为 n 的整数数列 a1...an。定义函数 f(i) 代表数列中第 i 个元素之后第一个大于 ai 的元素的下标。若不存在，则 f(i)=0。试求出 f(1...n)。

```python
n = int(input())
nums = list(map(int, input().split()))
stack = []
f = [0] * n
for i in range(n-1, -1, -1):
    while stack and nums[stack[-1]] <= nums[i]:
        stack.pop()
    if stack:
        f[i] = stack[-1] + 1
    stack.append(i)
# print(' '.join(map(str, f)))
print(*f)
```

### 1.4 辅助栈

22067：快速堆猪

```python
import heapq
from collections import defaultdict

pigs_heap = []
pigs_stack = []
out = defaultdict(int)

while True:
    try:
        c = input().split()
        if c[0] == 'pop':
            if pigs_stack:
                out[pigs_stack.pop()] += 1
        elif c[0] == 'push':
            pigs_stack.append(c[1])
            heapq.heappush(pigs_heap, c[1])
        elif c[0] == 'min':
            if pigs_stack:
                while True:
                    x = heapq.heappop(pigs_heap)
                    # 如果还没有被弹出来过，把它再放回去，跳出循环
                    if not out[x]:
                        heapq.heappush(pigs_heap, x)
                        print(x)
                        break
                    out[x] -= 1      
    except EOFError:
        break
```

## 2. 队列

collections库中的deque是双向队列，可以像普通列表一样访问，且在两端进出，复杂度都是O(1)

```python
import collections
dq = collections.deque()
dq.append('a') #右边入队
dq.appendleft(2) #左边入队
dq.extend([100,200]) #右边加入100,200
dq.extendleft(['c','d']) #左边依次加入 'c','d'
print(dq.pop()) #>>200 右边出队
print(dq.popleft()) #>>d 左边出队
print(dq.count('a')) #>>1
dq.remove('c') 
print(dq)	#>>deque([2, 'a', 100])
dq.reverse() 
print(dq)	#>>deque([100, 'a', 2])
print(dq[0],dq[-1],dq[1]) #>>100 2 a
print(len(dq)) #>>3

```

### 2.1 根据后序表达式建立队列表达式

```python
from collections import deque

class Node:
    def __init__(self, x):
        self.value = x
        self.left = None
        self.right = None

def build_tree(postfix):
    stack = []
    for item in postfix:
        if item in op:
            node = Node(item)
            node.right = stack.pop()
            node.left = stack.pop()
        else:
            node = Node(item)
        stack.append(node)
    return stack[0]

def dequeExp(root):
    if root is None:
        return

    stack = [] #用栈先进后出的特性，将树中的元素依次从根节点到叶结点依次存入
    queue = deque() #通过队列中转，先进先出
    queue.append(root)

    while queue:
        node = queue.popleft()
        stack.append(node)

        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    values = [node.value for node in stack[::-1]]
    print("".join(values))

num = 'abcdefghijklmnopqrstuvwxyz'
op = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

n = int(input())
for i in range(n):
    seq = list(input())
    root = build_tree(seq)
    dequeExp(root)
```

## 3. 树

### 3.1 根据二叉树中后序序列建树(24750)/根据二叉树前中序序列建树(22158)

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def buildTree(inorder, postorder):
    if not inorder or not postorder:
        return None
    
    root_val = postorder.pop()
    root = TreeNode(root_val)
    root_index = inorder.index(root_val) #找到根结点在中序遍历中的位置
    
    #将序列切分成右子树和左子树
    root.right = buildTree(inorder[root_index+1:], postorder) #给定中后序序列，应先遍历右子树再遍历左子树
    root.left = buildTree(inorder[:root_index], postorder)
    
    return root

def preorderTraversal(root):
    if not root:
        return []
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result

inorder = input().strip()
postorder = input().strip()

# 构建二叉树
root = buildTree(list(inorder), list(postorder))

# 前序遍历并输出结果
result = preorderTraversal(root)
print(''.join(result))
```



```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
    
    root_val = preorder.pop()
    root = TreeNode(root_val)
    root_index = inorder.index(root_val) #找到根结点在中序遍历中的位置
    
    #将序列切分成右子树和左子树
    root.left = buildTree(preorder, inorder[:root_index]) #给定前中序序列，应先遍历左树再遍历右树
    root.right = buildTree(preorder, inorder[root_index+1:])
    
    return root

def postorderTraversal(root):
    result = []
    if root:
        result.extend(postorderTraversal(root.left))
        result.extend(postorderTraversal(root.right))
        result.append(root.val)

    return result

while True:
    try:   
        preorder = input().strip()
        inorder = input().strip()

        # 构建二叉树
        root = buildTree(list(preorder)[::-1], list(inorder)) #此处需要反转输出前序序列！

        # 前序遍历并输出结果
        result = postorderTraversal(root)
        print(''.join(result))
    except:
        break
```

### 3.2 多叉树转换为二叉树

```python
class MultiTreeNode:
    def __init__(self, val):
        self.val = val
        self.children = []

class BinaryTreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def multi_to_binary(root):
    if not root:
        return None

    # 创建当前节点的二叉树节点
    binary_root = BinaryTreeNode(root.val)

    # 将第一个子节点作为左子节点
    if root.children:
        binary_root.left = multi_to_binary(root.children[0])

    # 处理其余子节点，将它们链接到右子树
    current = binary_root.left
    for child in root.children[1:]:
        current.right = multi_to_binary(child)
        current = current.right

    return binary_root
```

* 多叉树转二叉树，输出树的高度

```python
def height(s):
    max_old = 0
    max_new = 0
    old_height = 0
    new_height = 0
    stack = []
    for c in s:
        if c == 'd':
            old_height += 1
            max_old = max(max_old, old_height)

            new_height += 1
            stack.append(new_height)
            max_new = max(max_new, new_height)
        
        else:
            old_height -= 1

            new_height = stack.pop()
    return max_old, max_new

s = input()
old, new = height(s)
print('{}'.format(old) + ' => ' + '{}'.format(new))
```

### 3.3 最大堆最小堆实现

```python
import heapq
### 最大堆
class MaxHeap:
    def __init__(self):
        self.heap = []

    def push(self, val):
        heapq.heappush(self.heap, -val)  # 将元素取负值插入堆中

    def pop(self):
        return -heapq.heappop(self.heap)  # 弹出元素时取负值返回
### 最小堆（heapq)默认最小堆
x = [1,2,3,5,7]
heapq.heapify(x)
###将列表转换为堆。
heapq.heappushpop(heap, item)
##将 item 放入堆中，然后弹出并返回 heap 的最小元素。该组合操作比先调用 heappush() 再调用 heappop() 运行起来更有效率
heapq.heapreplace(heap, item)
##弹出并返回最小的元素，并且添加一个新元素item
heapq.heappop(heap,item)
heapq.heappush(heap,item)
```

### 3.4 bfs/dfs

```python
### 马走日(dfs)
def move(x, y, env, d, n, m):
    if x < 0 or x >= n or y < 0 or y >= m:
        return 0
    if env[x][y] != 0:
        return 0
    if d == n*m - 1:
        return 1
    
    env[x][y] = 1
    count = 0
    for deltax, deltay in actions:
        count += move(x + deltax, y + deltay, env, d + 1, n, m)
    env[x][y] = 0 #回溯，便于其他路径可以访问该状态
    return count

T = int(input())
actions = [[1, 2], [-1, 2], [1, -2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]
for i in range(T):
    n, m, x, y = map(int, input().split())
    env = [[0]*m for _ in range(n)]
    result = move(x, y, env, 0, n, m)
    print(result)
```



```python
### 词梯(bfs)
from collections import deque, defaultdict

def build_graph(words):
    graph = defaultdict(list)
    length = len(words[0]) # 单词长度都是4
    for word in words:
        for i in range(length):
            pattern = word[:i] + '*' + word[i+1:]
            graph[pattern].append(word)
    return graph

def bfs(start, end, graph):
    queue = deque([(start, [start])]) #初始化，用于存储当前的单词和途径的路径，同时队列先进先出的特性保证了bfs
    visited = set([start]) #记录访问过的单词（集合效率高）
    while queue:
        current_word, path = queue.popleft()
        if current_word == end:
            return path
        for i in range(len(current_word)):
            pattern = current_word[:i] + '*' + current_word[i+1:]
            for neighbor in graph[pattern]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
    return [] #没找到要返回空！

def word_ladder(start, end, words):
    graph = build_graph(words)
    return bfs(start, end, graph)

n = int(input())
words = []
for i in range(n):
    words.append(input())
start, end = input().split()
result = word_ladder(start, end, words)
if len(result) == 0:
    print('NO')
else:
    print(' '.join([_ for _ in result]))
```

### 3.6 二叉搜索树的遍历

```python
def preorder_to_postorder(preorder):
    if not preorder:
        return []
    
    root = preorder[0]
    left_preorder = [x for x in preorder[1:] if x < root]
    right_preorder = [x for x in preorder[1:] if x > root]
    
    left_postorder = preorder_to_postorder(left_preorder)
    right_postorder = preorder_to_postorder(right_preorder)
    
    return left_postorder + right_postorder + [root]

n = int(input())
preorder = list(map(int, input().split()))
postorder = preorder_to_postorder(preorder)
print(' '.join(map(str, postorder)))
```

### 3.7 二叉搜索树的层次遍历

```python
from collections import deque

numbers = list(map(int, input().strip().split()))
numbers = list(dict.fromkeys(numbers)) # remove duplicates

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 永远从根开始往下插入，遇到比当前根节点值小的往左插，否则往右插
def insert(node, value):
    if node is None:
        return TreeNode(value)
    if value < node.val:
        node.left = insert(node.left, value)
    else:
        node.right = insert(node.right, value)
    return node

def traversal(root):
    queue = deque([root])
    travesal = []
    while queue:
        node = queue.popleft()
        travesal.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return travesal

root = None
for number in numbers:
    root = insert(root, number)
traversal = traversal(root)
print(' '.join(map(str, traversal)))
```

## 4. 图

### 4.1 dijkstra算法

1. 初始化：
   - 将起点的距离设为0，其余所有节点的距离设为无穷大（表示尚未访问）。
   - 使用一个优先队列（通常是最小堆）来存储节点及其当前已知的最短距离。
   - 使用一个集合来记录已确定最短路径的节点。
2. 选取当前最小距离的节点：
   - 从优先队列中提取当前距离最小的节点作为当前节点。
3. 更新邻居节点的距离：
   - 对当前节点的每个邻居节点，计算从起点经过当前节点到邻居节点的距离。
   - 如果这个距离小于邻居节点当前已知的最短距离，则更新邻居节点的最短距离，并将该邻居节点添加到优先队列中。
4. 重复步骤2和3，直到所有节点都被处理或优先队列为空。

```python
import heapq
def dijkstra(graph, start):
    # 初始化距离字典，所有节点的初始距离为无穷大
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    # 优先队列，存储 (距离, 节点)
    priority_queue = [(0, start)]
    heapq.heapify(priority_queue)
    # 记录已确定最短路径的节点
    visited = set()
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_node in visited:
            continue
        visited.add(current_node)
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            # 如果找到更短的路径，更新邻居节点的距离
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances
```

```python
### 兔子与樱花（求路径最短走法）
import heapq

def dijkstra(graph, start, end):
    if start == end:
        return f"{start}"

    # 初始化距离字典和路径字典
    dist = {vertex: float('inf') for vertex in graph}
    dist[start] = 0
    path = {vertex: ([]) for vertex in graph}
    path[start] = [(start, 0)]  # 起点没有前一个节点，距离为0

    # 优先队列，用于存储待处理的顶点和距离
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))

    while priority_queue:
        current_dist, current_vertex = heapq.heappop(priority_queue)

        # 遍历当前顶点的邻接顶点
        for neighbor, weight in graph[current_vertex].items():
            distance = current_dist + weight

            # 如果找到更短的路径，则更新
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                path[neighbor] = path[current_vertex] + [(neighbor, weight)]
                heapq.heappush(priority_queue, (distance, neighbor))

        # 如果当前顶点是终点，格式化输出路径
        if current_vertex == end:
            return format_path(path[end])

    # 如果终点不可达，返回空字符串
    return ""

def format_path(path):
    if not path:
        return ""
    formatted_path = path[0][0]  # 起始节点
    for vertex, weight in path[1:]:
        formatted_path += f"->({weight})->{vertex}"
    return formatted_path

P = int(input())
graph = {} 
graph = {input(): {} for _ in range(P)}

Q = int(input())
for i in range(Q):
    start, end, cost = input().split()
    graph[start][end] = graph[end][start] = int(cost)

R = int(input())
for i in range(R):
    start, end = input().split()
    path = dijkstra(graph, start, end)
    print(path)
```

### 4.2 Prim算法

```python
from collections import defaultdict
from heapq import *
def prim(vertexs, edges,start='D'):
    adjacent_dict = defaultdict(list) # 注意：defaultdict(list)必须以list做为变量
    for weight,v1, v2 in edges:
        adjacent_dict[v1].append((weight, v1, v2))
        adjacent_dict[v2].append((weight, v2, v1))
    minu_tree = []  # 存储最小生成树结果
    visited = [start] # 存储访问过的顶点，注意指定起始点
    adjacent_vertexs_edges = adjacent_dict[start]
    heapify(adjacent_vertexs_edges) # 转化为小顶堆，便于找到权重最小的边
    while adjacent_vertexs_edges:
        weight, v1, v2 = heappop(adjacent_vertexs_edges) # 权重最小的边，并同时从堆中删除。 
        if v2 not in visited:
            visited.append(v2)  # 在used中有第一选定的点'A'，上面得到了距离A点最近的点'D',举例是5。将'd'追加到used中
            minu_tree.append((weight, v1, v2))
            # 再找与d相邻的点，如果没有在heap中，则应用heappush压入堆内，以加入排序行列
            for next_edge in adjacent_dict[v2]: # 找到v2相邻的边
                if next_edge[2] not in visited: # 如果v2还未被访问过，就加入堆中
                    heappush(adjacent_vertexs_edges, next_edge)
    return minu_tree
```

```python
### 兔子与星空（解决最小生成树问题）
import heapq

def prim(graph, start):
    mst = []
    used = set([start])
    edges = [
        (cost, start, to)
        for to, cost in graph[start].items()
    ]
    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in used:
            used.add(to)
            mst.append((frm, to, cost))
            for to_next, cost2 in graph[to].items():
                if to_next not in used:
                    heapq.heappush(edges, (cost2, to, to_next))

    return mst

n = int(input())
graph = {chr(i+65): {} for i in range(n)} # ASCII encoding
for i in range(n-1):
    data = input().split()
    star = data[0]
    m = int(data[1]) # how many stars is connected to current star
    for j in range(m):
        to_star = data[2+j*2]
        cost = int(data[3+j*2])
        graph[star][to_star] = cost
        graph[to_star][star] = cost
mst = prim(graph, 'A')
print(sum(x[2] for x in mst))
```

### 4.3 Kruskal算法

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def kruskal(graph):
    edges = []
    for u in range(len(graph)):
        for v, weight in enumerate(graph[u]):
            if u < v and weight != 0:
                edges.append((weight, u, v))
    edges.sort()

    uf = UnionFind(len(graph))
    mst_cost = 0
    mst_edges = []

    for weight, u, v in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst_cost += weight
            mst_edges.append((u, v))

    return mst_cost, mst_edges
```

### 4.4 判断无向图连通性和是否成环

```python
def isConnected(G): # G 是邻接表,顶点编号从 0 开始，判断是否连通
     n = len(G)
     visited = [False for _ in range(n)]
     total = 0

     def dfs(v):
         nonlocal total
         visited[v] = True
         total += 1
         for u in G[v]:
             if not visited[u]:
                 dfs(u)
     dfs(0)
     return total == n

def hasLoop(G): # G 是邻接表,顶点编号从 0 开始，判断有无回路
     n = len(G)
     visited = [False for _ in range(n)]

     def dfs(v, x): # 返回值表示本次 dfs 是否找到回路,x 是深度优先搜索树上 v 的父结点
         visited[v] = True
         for u in G[v]:
             if visited[u] == True: # 如果 u 已经访问过
                 if u != x: # u 不是 v 的父结点
                     return True
             else: # 如果 u 没有访问过
                 if dfs(u, v): # 递归调用 dfs
                     return True
         return False

     for i in range(n):
         if not visited[i]:
             if dfs(i, -1):
                 return True
     return False

n, m = map(int, input().split())
G = [[] for _ in range(n)]
for _ in range(m):
     u, v = map(int, input().split())
     G[u].append(v)
     G[v].append(u)
```

### 4.5 有向图判断（弱）连通性（拓扑排序）

```python
from collections import defaultdict, deque

def is_weakly_connected(graph):
    # Convert directed graph to undirected graph
    undirected_graph = defaultdict(set)
    for u in graph:
        for v in graph[u]:
            undirected_graph[u].add(v)
            undirected_graph[v].add(u)

    # BFS or DFS to check if all nodes are reachable
    visited = set()
    start_node = next(iter(undirected_graph))
    queue = deque([start_node])
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(undirected_graph[node] - visited)
    
    return len(visited) == len(undirected_graph)

# Example usage
graph = {
    'A': ['B'],
    'B': ['C'],
    'C': ['A', 'D'],
    'D': []
}

print(is_weakly_connected(graph))  # Output: True or False depending on graph structure
```

### 4.6 并查集

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            # Union by rank
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

    def connected(self, u, v):
        return self.find(u) == self.find(v)
      
    def judge(self):
        # Ensure all nodes are updated to their root
        for i in range(len(self.parent)):
            self.find(i)
        return len(set(self.parent))

# Example usage
if __name__ == "__main__":
    # Initialize UnionFind with 5 elements (0 through 4)
    uf = UnionFind(5)
    
    # Union operations
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)

    # Find operations
    print(uf.find(0))  # Output: root of 0, which will be same as root of 1 and 2
    print(uf.find(1))  # Output: root of 1
    print(uf.find(2))  # Output: root of 2
    print(uf.find(3))  # Output: root of 3, which will be same as root of 4
    print(uf.find(4))  # Output: root of 4

    # Connected operations
    print(uf.connected(0, 2))  # Output: True
    print(uf.connected(0, 3))  # Output: False
```

## 5. 回溯

```python
### 八皇后
ans = []
def queen_dfs(A, cur=0):
    if cur == len(A):
        ans.append(''.join([str(x+1) for x in A]))
        return
    for col in range(len(A)): 
        for row in range(cur):
            # 检查之前的行是否有皇后在同一列或在同一对角线上
            if A[row] == col or abs(col - A[row]) == cur - row:
                break
        else:
            A[cur] = col #对应行的皇后所在的列
            queen_dfs(A, cur+1)

queen_dfs([None]*8)

for _ in range(int(input())):
    print(ans[int(input()) - 1])
```

