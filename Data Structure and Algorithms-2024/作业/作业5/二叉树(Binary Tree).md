# 二叉树(Binary Tree)

## 二叉树概念及性质

1. 二叉树是有限个元素的集合
2. 空集合是一个二叉树，称为空二叉树
3. 一个元素（称其为“根”或“根结点”），加上一个被称为“左子树”的二叉树，和一个被称为“右子树”的二叉树，就能形成一个新的二叉树。**要求根、左子树和右子树三者没有公共元素**

<img src="/Users/lijialin/Library/Application Support/typora-user-images/Screenshot 2024-03-20 at 3.46.45 PM.png" alt="Screenshot 2024-03-20 at 3.46.45 PM" style="zoom: 33%;" />

4. 二叉树的左右子树是有区别的

   <img src="/Users/lijialin/Library/Application Support/typora-user-images/Screenshot 2024-03-20 at 3.48.50 PM.png" alt="Screenshot 2024-03-20 at 3.48.50 PM" style="zoom: 33%;" /><img src="/Users/lijialin/Library/Application Support/typora-user-images/Screenshot 2024-03-20 at 3.49.10 PM.png" alt="Screenshot 2024-03-20 at 3.49.10 PM" style="zoom: 33%;" />







### 二叉树的相关概念

* 二叉树的的元素称为“结点”。结点由三部分组成：数据、左子结点指针、右子结点指针。

* 结点的度(degree)：结点的非空子树数目。也可以说是结点的子结点数目。

* **叶结点(leaf node)：度为0的结点。**

* **分支结点：度不为0的结点**。即除叶子以外的其他结点。也叫内部结点。

* 兄弟结点(sibling)：父结点相同的两个结点，互为兄弟结点。

* 结点的层次(level)：树根是第0层的。如果一个结点是第n层的，则其子结点就是第n+1层的。

* 结点的深度(depth)：即结点的层次

* 祖先(ancestor):
  * 父结点是子结点的祖先
  * 若a是b的祖先，b是c的祖先，则a是c的祖先。

* 子孙(descendant)：也叫后代。若结点a是结点b的祖先，则结点b就是结点a的后代。

* 边：若a是b的父结点，则对子<a,b>就是a到b的边。在图上表现为连接父结点和子结点之间的线段。

* 二叉树的高度(height)：二叉树的高度就是结点的最大层次数。只有一个结点的二叉树，高度是0。结点一共有n层，高度就是n-1。
* 完美二叉树：每一层结点数目都达到最大。即第i层有$$2^i$$个结点。高为h的完美二叉树，有$$2^{h+1}-1$$个结点

<img src="/Users/lijialin/Library/Application Support/typora-user-images/image-20240320155628581.png" alt="image-20240320155628581" style="zoom: 50%;" />

* 满二叉树(full binary tree)：没有1度结点的二叉树

<img src="/Users/lijialin/Library/Application Support/typora-user-images/image-20240320155653762.png" alt="image-20240320155653762" style="zoom:50%;" />

* 完全二叉树(complete binary tree)：除最后一层外，其余层的结点数目均达到最大。而且，最后一层结点若不满，则缺的结点定是在最右边的连续若干个

<img src="/Users/lijialin/Library/Application Support/typora-user-images/image-20240320155808933.png" alt="image-20240320155808933" style="zoom:50%;" />

### 二叉树的性质

* 第i层最个多2i个结点

* 高为h的二叉树结点总数最多2h+1-1

* 结点数为n的树，边的数目为n-1

* n个结点的非空二叉树至少有⌈log2(n+1)⌉层结点，即高度至少为⌈log2(n+1)⌉- 1

  > 假设我们有 n 个节点，那么当我们达到或超过 n 时，树的高度将是我们的目标高度。考虑到二叉树每一层的节点数量都是 2 的指数增长，我们可以找到最小的 k，使得 2^(k-1) >= n，即：
  >
  > 2^(k-1) >= n
  >
  > 然后，通过对上述不等式取对数，我们可以得到：
  >
  > log2(2^(k-1)) >= log2(n)
  >
  > k - 1 >= log2(n)
  >
  > k >= log2(n) + 1
  >
  > 由于 k 必须是整数，我们向上取整到最接近的整数，因此我们得到：
  >
  > k = ⌈log2(n)⌉ + 1
  >
  > 但是因为我们是从第二层开始计数的，所以实际的高度应该是 k - 1：
  >
  > 实际高度 = ⌈log2(n)⌉

* 在任意一棵二叉树中，若叶子结点的个数为n0，度为2的结点个数为n2，则n0=n2+1。

* 非空满二叉树叶结点数目等于分支结点数目加1。

* 非空二叉树中的空子树数目等于其结点数目加1。

### 完全二叉树的性质

* 完全二叉树中的1度结点数目为0个或1个

* 有n个结点的完全二叉树有[(n+1)/2]个叶结点。

* 有n个叶结点的完全二叉树有2n或2n-1个结点(两种都可以构建)

* 有n个结点的非空完全二叉树的高度为⌈log2(n+1)⌉-1。即：有n个结点的非空完全二叉树共有⌈log2(n+1)⌉层结点。

* 可以用列表存放完全二叉树的结点，不需要左右子结点指针。下标为i的结点的左子结点下标是$$2^i+1$$,右子结点是$$2^i+2$$ (根下标为0)。下标为i的元素，其父结点的下标就是(i-1)//2 

## 二叉树的实现

