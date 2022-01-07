# 1.数组中重复的数字-数组-简单

> 题目

找出数组中重复的数字。


在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

> 思路：注意大小不超过数组下标

没有重复值意味着每个数都刚好能等于数组下标，如果当前数不等于数组下标就与当前值应该存在的下标位置的数进行交换，直到交换到当前值等于数组下标

> 代码

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
        int i = 0;
        while(i < nums.length){
            if(nums[i] == i){
                i++;
                //注意相等就直接跳过这次循环
                continue;
            }
            if(nums[nums[i]] == nums[i]){
                return nums[i];
            }
            int temp = nums[i];
            nums[i] = nums[nums[i]];
            nums[temp] = temp;
        }
        return -1;
    }
}
```

# 2.二维数组中的查找-数组-中等

> 题目

在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

> 思路：从左下角或右上角的特殊性考虑

左下角或右上角的数都比左边大右边小

> 代码

```java
class Solution {
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
      int row = matrix.length;
      if(row == 0) return false;
      int col = matrix[0].length;
      int i = 0, j = col - 1;
      while(i < row && j >= 0){
          if(matrix[i][j] == target){
              return true;
          }
          if(matrix[i][j] > target){
              j--;
          }else{
              i++;
          }
      }
      return false;
    }
}
```

# 3.替换空格-字符串-简单

> 题目

请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

> 思路：没感觉到这题想考些什么，new String(arr, 0, size)？

官方题解：新建一个大小为字符串长度3倍的char数组，如果不是空格就将当前字符填充到char数组对应位置，是空格就填充%20.最后使用`new String(charArray, 0, size)`

> 代码

```java
class Solution {
    public String replaceSpace(String s) {
        int length = s.length();
        char[] array = new char[length * 3];
        int size = 0;
        for (int i = 0; i < length; i++) {
            char c = s.charAt(i);
            if (c == ' ') {
                array[size++] = '%';
                array[size++] = '2';
                array[size++] = '0';
            } else {
                array[size++] = c;
            }
        }
        String newStr = new String(array, 0, size);
        return newStr;
    }
}
```



# 4. 从尾到头打印链表-链表-简单

> 题目

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

> 思路：栈或递归自己选

> 代码

```java
class Solution {
    public int[] reversePrint(ListNode head) {
        Deque<Integer> stack = new ArrayDeque<>();
        while(head != null){
            stack.push(head.val);
            head = head.next;
        }
        int[] res = new int[stack.size()];
        for(int i = 0; i < res.length; i++){
            res[i] = stack.pop();
        }
        return res;
    }
}
```

# 5. 重建二叉树-二叉树-中等

> 题目

输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

> 思路：与leetcode 105题一样

> 代码

```java
class Solution {
    HashMap<Integer, Integer> inorderMap = new HashMap<>();
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if(preorder.length == 0) return null;
        //初始化map，方便后续快速查找到某个值在中序列表中的下标
        for(int i = 0; i < inorder.length; i++){
            inorderMap.put(inorder[i], i);
        }
        TreeNode root = buildTree(preorder, inorder, 0, preorder.length - 1, 0, inorder.length - 1);
        return root;
    }
    public TreeNode buildTree(int[] preorder, int[] inorder, int preorderStart, int preorderEnd, int inorderStart, int inorderEnd) {
        if(inorderStart > inorderEnd) return null;
        TreeNode root = new TreeNode(preorder[preorderStart]);
        //得到根结点元素在中序列表里的下标位置
        int index = inorderMap.get(preorder[preorderStart]);
        //计算左子树结点个个数，确定左右子树前序遍历列表区间和中序遍历列表区间
        int count = index - inorderStart;
        root.left = buildTree(preorder, inorder, preorderStart + 1, preorderStart + count, inorderStart, index - 1);
        root.right = buildTree(preorder, inorder, preorderStart + count + 1, preorderEnd, index + 1, inorderEnd);
        return root;
    }
}
```

# 6. 用两个栈实现队列]

> 题目

用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )



> 思路

1、取数据时`如果取数据的栈为空`就把插入数据的栈依次出栈放入取数据的栈中，这样取数据时就保证是保存插入数据的栈低元素，也就是最先插入进来的数据不为空直接就pop

2、插入数据直接向保存插入数据的栈中直接插入即可。

> 代码

```java
class CQueue {

    Deque<Integer> stack_in;
    Deque<Integer> stack_out;
    public CQueue() {
        stack_in = new ArrayDeque<>();
        stack_out = new ArrayDeque<>();
    }
    
    public void appendTail(int value) {
       stack_in.push(value);
    }
    
    public int deleteHead() {
       if(!stack_out.isEmpty()){
           return stack_out.pop();
       }else{
           //将入栈数据依次出栈到out栈中
           while(!stack_in.isEmpty()){
               stack_out.push(stack_in.pop());
           }
       }
       return stack_out.isEmpty() ? -1 : stack_out.pop();
    }
}

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue obj = new CQueue();
 * obj.appendTail(value);
 * int param_2 = obj.deleteHead();
 */
```

# 7. 斐波那契数列

> 题目

写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：

F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

> 思路：动态规划

> 代码

```java
class Solution {
    public int fib(int n) {
        int pre1 = 1, pre2 = 0, res = 1;
        if(n == 0) res = pre2;
        if(n == 1) res = pre1;
        for(int i = 2; i <= n; i++){
            res = (pre1 + pre2)%1000000007;
            pre2 = pre1;
            pre1 = res;
        }
        return res%1000000007;
    }
}
```

# 8. 青蛙跳台阶问题

> 题目

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

> 思路：动态规划

> 代码

```java
class Solution {
    public int numWays(int n) {
        int pre1 = 2, pre2 = 1, res = 0;
        if(n <= 1) res = 1;
        if(n == 2) res = 2;
        for(int i = 3; i <= n; i++){
            res = (pre2 + pre1) % 1000000007; 
            pre2 = pre1;
            pre1 = res;
        }
        return res;
    }
}
```

# 9. 旋转数组的最小数字

> 题目

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。

> 思路：二分

中间位置和右边位置元素比较大小，由于是旋转数组，所以：

- 如果中间的元素大于最右边的，那么最小值一定在右边

- 与最右边等于时无法二分，只能right--
- 比最右边小于时由于中间位置元素可能刚好是最小的，所以right=mid，而不是=mid-1

> 代码

```java
class Solution {
    public int minArray(int[] numbers) {
        int left = 0, right = numbers.length - 1, mid = 0;
        while(left < right){
             mid = left + (right - left) / 2;
            if(numbers[mid] > numbers[right]){
                left = mid + 1;
            }else if(numbers[mid] == numbers[right]){ 
                right--;
            }else{
                right = mid;
            }
        }
        return numbers[left];
    }
}
```

# 10. 矩阵中的路径

> 题目

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。

[["a","b","c","e"],
["s","f","c","s"],
["a","d","e","e"]]

但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。

> 思路：深度优先搜索+剪枝

dfs：判断当前位置是否与对应位置字符相等，相等则往该位置的四个方向DFS

剪枝：判断该元素前先判断是否已经遍历过，遍历过直接返回false，否则在遍历其它方向时将其设置为已遍历过，遍历完剪枝：设置为未遍历

> 代码

```java
class Solution {
    int[][] flag;
    public boolean exist(char[][] board, String word) {
        if(word.length() == 0) return false;
        flag = new int[board.length][board[0].length];
        for(int i = 0; i < board.length; i++){
            for(int j = 0; j < board[0].length; j++){
                if(exist(board, i, j, word, 0)){
                    return true;
                }
            }
        }
        return false;
    }
    public boolean exist(char[][] board, int i, int j, String word, int index) {
        if(flag[i][j] == 1) return false;
        if(board[i][j] == word.charAt(index)){
            if(index == word.length() - 1){
                return true;
            }
            //设置为已经遍历过（为true直接就返回了，不用管剪枝了）
            flag[i][j] = 1;
            if(i - 1 >= 0){
                if(exist(board, i - 1, j, word, index + 1)) return true;
            }
            if(i + 1 < board.length){
                if(exist(board, i + 1, j, word, index + 1)) return true;
            }
            if(j - 1 >= 0){
                if(exist(board, i, j - 1, word, index + 1))return true;
            }
            if(j + 1 < board[0].length){
                if(exist(board, i, j + 1, word, index + 1))return true;
            }
            //剪枝
            flag[i][j] = 0;
            return false;
        }else{
            return false;
        }
    }
}
```

# 11. 机器人的运动范围

> 题目

地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

 

> 思路：深度优先搜索

不符合条件的格子直接返回

符合条件的话就将总格子数加1，并将该格子标记为已经走过，继续遍历其它方向（只需DFS向下和向右两个方向，不需要向四个方向遍历）

> 代码

```java
class Solution {
    int res = 0;
    boolean[][] flag;
    public int movingCount(int m, int n, int k) {
        flag = new boolean[m][n];
        movingCount(0, 0, m, n, k);
        return res;
    }
    public void movingCount(int i, int j, int m, int n, int k) {
       if(i >= m || j >= n || flag[i][j] || getSum(i, j) > k) return;
       res++;
       flag[i][j] = true;
       movingCount(i + 1, j, m, n, k);
       movingCount(i, j + 1, m, n, k);
    }
    public int getSum(int i, int j){
        int sum = 0;
        while(i > 0){
            sum = sum + i % 10;
            i /= 10;
        }
        while(j > 0){
            sum = sum + j % 10;
            j /= 10;
        }
        return sum;
    }
}
```

# 12. 剪绳子

> 题目

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

 

> 思路1：每一段接近3乘积最大

> 代码

```java
class Solution {
    public int cuttingRope(int n) {
       int t = n % 3;
       int cnt = n / 3;
       if(n <= 3) return n - 1;
       if(t == 1) {
           return (int)Math.pow(3, cnt - 1) * 4;
       }
       if(t == 2) {
           return (int)Math.pow(3, cnt) * 2;
       }
       return (int)Math.pow(3, cnt);
    }
}
```

> 思路2：动态规划

> 代码

```java
class Solution {
    public int cuttingRope(int n) {
       int[] dp = new int[n + 1];
       dp[2] = 1;
       dp[3] = 2;
       for(int i = 4; i <= n; i++){
           for(int j = 2; j <= i - 2; j++){
               //从2开始划分，每次划分后面长度可以划分也可以不划分，选乘积最大的即可。
               dp[i] = Math.max(dp[i], Math.max(j * (i - j), j * dp[i - j]));
           }
       }
       return dp[n];
    }
}
```

# 13. 剪绳子2

> 题目

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

 

> 思路：动态规划取余将数组设置为long类型。这里使用每一段接近3乘积最大原理

```java
class Solution {
    public int cuttingRope(int n) {
        if(n < 4) return n - 1;
        int p = 1000000007;
        long res = 1L;
        //为4就不继续乘3了，乘4更大。最后剩下的数为:2、3、4
        while(n > 4){
            //每一次计算都要取余
            res = (res * 3) % p;
            n -= 3;
        }
        //乘以2、3或4取余即可
        return (int)(n * res % p);
    }
}
```

# 14.二进制中1的个数

> 题目

请实现一个函数，输入一个整数（以二进制串形式），输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。

> 思路1：常规计算二进制位，主要考虑负数

负数的的话需要将二进制位第一个1换为0，然后将换了之后表示的整数再进行操作，最后结果记得加1

正数直接常规算。

> 代码

```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int cnt = 0;
        if(n < 0){
            //进行异或运算：相同为0，不同为1，这样就可以将负数二进制的符号位变位0了
            n = n ^ (1 << 31);
            cnt++;//记得加1
        }
        while(n > 0){
            //n%2即是最低位的二进制位，最后反过来就是二进制数，这里只用管是不是1不用反
            if(n % 2 == 1) cnt++;
            n /= 2;
        }
        return cnt;
    }
}
```

> 思路2：与1进行按位与运算

如果最右边以为是0，与运算结果就为0否则为1，每次运算完原来的数就向右移动一位直至原数为0表示移动完了

`注意使用无符号右移`

> 代码

```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int res = 0;
        while(n != 0){
            if((n & 1) == 1){
                res++;
            }
            n >>>= 1;
        }
        return res;
    }
}
```

# 15.数值的整数次方

> 题目

实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题

> 思路：快速幂

将指数m转成二进制数：b1b2b3...bn，再将这二进制数转成十进制：b1乘以2^0 + b2乘以2^1 + b3乘以2^2 +......+ bn乘以2^(n-1)。则原来的式子可以转化成X^(m) = X^(b1乘以2^0 + b2乘以2^1 + b3乘以2^2 +......+ bn乘以2^(n-1)) = X^(b1乘以2^0) * X^( b2乘以2^1 ).........


 转化为解决以下两个问题：

计算 x^1, x^2, x^4, ..., x^{2^{n-1}}x 
获取二进制各位 b1,b2,b3....bn

> 代码

```java
class Solution {
    public double myPow(double x, int n) {
        double res = 1;
        //如果是负数最小值转化为正数int会超出范围
        long temp = n;
        if(temp < 0){
            x = 1 / x;
            temp *= -1;
        }
        while(temp != 0){
            if((temp & 1) == 1){
                res *= x;
            }
            x *= x;
            temp >>= 1;
        }
        return res;
    }
}
```

# 16.打印从1到最大的n位数-简单

> 题目

输入数字 `n`，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

**示例 1:**

```txt
输入: n = 1
输出: [1,2,3,4,5,6,7,8,9]
```

> 思路：全排列，递归回溯

递归出口：回溯到数字第n位时就处理结果，该数的回溯递归结束

从0-9这10个字符串中进行排列组合，每一位都有10种字符可能

> 代码

```java
class Solution {
    ArrayList<String> list = new ArrayList<>();
    public int[] printNumbers(int n) {
        printNumbers(n, 0, "");
        int[] res = new int[list.size() - 1];
        for(int i = 1; i < list.size(); i++){
            res[i - 1] = Integer.parseInt(list.get(i), 10);
        }
        return res;
    }
    public void printNumbers(int n, int count, String res) {
        if(count == n){
            list.add(res);
            return ;
        }
        for(int i = 0; i < 10; i++){
            printNumbers(n, count + 1, res + i);
        }
    }
}
```

# 17.删除链表的节点-简单

> 题目

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点。

> 思路：递归、前后指针迭代都行

递归出口：当前节点值等于目标值或当前节点为null

> 代码

```java
class Solution {
    public ListNode deleteNode(ListNode head, int val) {
        if(head == null) return null;
        if(head.val == val) return head.next;
        head.next = deleteNode(head.next, val);
        return head;
    }
}
```

# 18.正则表达式匹配-困难

> 题目

请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。

> 思路：动态规划（自顶向下）

> 代码

```java
class Solution {
    public boolean isMatch(String s, String p) {
        return isMatch(s, p, s.length() - 1, p.length() - 1);
    }
    public boolean isMatch(String s, String p, int sEnd, int pEnd) {
        if(pEnd < 0){
            if(sEnd >= 0) return false;
            else return true;
        }
        if(sEnd < 0){
            //不是字符+*就不匹配
            if(pEnd >= 0){
                for(int i = pEnd; i >= 0; i -= 2){
                    if(p.charAt(i) != '*') return false;
                }  
            }
            return true;
        }
        char pChar = p.charAt(pEnd);
        char sChar = s.charAt(sEnd);
        //不属于特殊符号*和.就直接对照匹配
        if(pChar != '*' && pChar != '.'){
            if(pChar != sChar) return false;
            else return isMatch(s, p, sEnd - 1, pEnd - 1);
        }else{
            //.匹配任意的
            if(pChar == '.'){
                return isMatch(s, p, sEnd - 1, pEnd - 1);
            }else{
                //*号：如果*号前面的字符与s对应字符匹配，则可以选择匹配任意数量个
                boolean temp = false;
                if(sChar == p.charAt(pEnd - 1) || p.charAt(pEnd - 1) == '.'){
                    temp = isMatch(s, p, sEnd - 1, pEnd);
                }
                //还可以跳过相当于0次匹配
                return temp || isMatch(s, p, sEnd, pEnd - 2);
            }
        }
    }
}
```

> 思路：动态规划（自底向上）

> 代码

```java
class Solution {
    public boolean isMatch(String s, String p) {
        //默认值全是false
        boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
        for(int i = 0; i < dp.length; i++){
            for(int j = 0; j < dp[0].length; j++){
                //匹配串为空:字符串不为空肯定就不匹配
                if(j == 0){
                    dp[i][j] = (i == 0);
                }else{
                    if(p.charAt(j - 1) == '*'){
                        //匹配0次,相当于直接跳过,此时j肯定大于等于2
                        dp[i][j] = dp[i][j - 2];
                        //*号前面的字符是.或者能等于s的对应字符才能匹配多次
                        if(i > 0 && (s.charAt(i - 1) == p.charAt(j - 2) || p.charAt(j - 2) == '.')){
                            dp[i][j] = dp[i][j] || dp[i - 1][j];
                        }
                    }else{
                       if(i > 0 && (p.charAt(j - 1) == '.' || p.charAt(j - 1) == s.charAt(i - 1))){
                           dp[i][j] = dp[i - 1][j - 1];
                       }
                    }
                }
            }
        }
        return dp[s.length()][p.length()];
    }
}
```

# 19.表示数值的字符串-困难

> 题目

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。

`数值（按顺序）可以分成以下几个部分：`

1. 若干空格
2. 一个 小数 或者 整数
3. （可选）一个 'e' 或 'E' ，后面跟着一个 整数
4. 若干空格

`小数（按顺序）可以分成以下几个部分：`

1. （可选）一个符号字符（'+' 或 '-'）
2. 下述格式之一：
   至少一位数字，后面跟着一个点 '.'
   至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
   一个点 '.' ，后面跟着至少一位数字



`整数（按顺序）可以分成以下几个部分：`

1. （可选）一个符号字符（'+' 或 '-'）
2. 至少一位数字



部分数值列举如下：

- ["+100", "5e2", "-123", "3.1416", "-1E-16", "0123"]
  部分非数值列举如下：

- ["12e", "1a3.14", "1.2.3", "+-5", "12e+5.4"]

> 思路：我是傻X



> 代码

# 20.调整数组顺序使奇数位于偶数前面-简单

> 题目

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

> 思路：和快排使得一边全小于某个数一边全大于某个数一样

定义一个指针指向分界点，默认为0，如果符合放在前半部分的条件就和该指针位置的元素进行交换，然后该指针往后移一位。

> 代码

```java
class Solution {
    public int[] exchange(int[] nums) {
        int index = 0;
        for(int i = 0; i < nums.length; i++){
            if(nums[i] & 1 != 0){
                int temp = nums[i];
                nums[i] = nums[index];
                nums[index] = temp;
                index++;
            }
        }
        return nums;
    }
}
```

# 21.链表中倒数第k个节点-简单

> 题目

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。

> 思路：双指针，遍历一遍即可

> 代码

```java
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        if(head == null) return null;
        //开始都指向第一个结点
        ListNode left = head, right = head;
        while(k > 1){
            right = right.next;
            k--;
        }
        while(right.next != null){
            left = left.next;
            right = right.next;
        }
        return left;
    }
}
```

# 22.反转链表-简单

> 题目

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

> 思路1：前后指针迭代

> 代码

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null) return null;
        ListNode left = head, right = head.next;
        while (right != null){
            ListNode temp = right.next;
            right.next = left;
            left = right;
            right = temp;
        }
        head.next = null;
        return left;
    }
}
```

> 思路2：递归

> 代码

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode newHead = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }
}
```

# 23.合并两个排序的链表-简单

> 题目

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

> 思路1：归并思想

> 代码

```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode tempHead = new ListNode(-1);
        ListNode temp = tempHead;
        while(l1 != null && l2 != null){
            if(l1.val <= l2.val){
                temp.next = l1;
                temp = l1;
                l1 = l1.next;
            }else{
                temp.next = l2;
                temp = l2;
                l2 = l2.next;
            }
        }
        temp.next = l1 == null ? l2 : l1;
        return tempHead.next;
    }
}
```

> 思路2：递归

> 代码

```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null) return l2;
        if(l2 == null) return l1;
        if(l1.val <= l2.val){
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        }else{
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }
}
```

# 24.树的子结构-中等

> 题目

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:

     3
    / \

   4   5
  / \
 1   2
给定的树 B：

   4 
  /
 1
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

> 思路：递归

遍历找到与子结构树根节点相等的节点，然后判断该节点与子结构根节点是否 “ 相等 ”

> 代码

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if(A == null || B == null) return false;
        if(A.val == B.val){
            //两个结点值相等才使用isSubStructure2
            return isSubStructure2(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B);
        }
        return isSubStructure(A.left, B) || isSubStructure(A.right, B);
    }
    public boolean isSubStructure2(TreeNode A, TreeNode B) {
        if(B == null) return true;//B结点为空代表匹配到这里已经结束了，返回true（才进入这函数时A与B一定都不为null且val相等）
        if(A == null) return false;//B不为null，A为null代表无法匹配
        if(A.val == B.val){
            //继续匹配双方的左右结点
            return isSubStructure2(A.left, B.left) && isSubStructure2(A.right, B.right);
        }else{
            return false;
        }
    }
}
```

# 25. 二叉树的镜像-简单

> 题目

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

例如输入：

     4

   /   \
  2     7
 / \   / \
1   3 6   9
镜像输出：

     4

   /   \
  7     2
 / \   / \
9   6 3   1

> 思路：递归

> 代码

```java
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if(root == null) return null;
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        mirrorTree(root.left);
        mirrorTree(root.right);
        return root;
    }
}
```

# 26. 对称的二叉树-简单

> 题目

```java
请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

`例如，二叉树 [1,2,2,3,4,4,3] 是对称的。`

1

   / \
  2   2
 / \ / \
3  4 4  3
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

1

   / \
  2   2
   \   \
   3    3
```

> 思路：递归

对称二叉树定义： 对于树中 任意两个对称节点 LL 和 RR ，一定有：

- L.val = R.valL.val=R.val ：即此两对称节点值相等。
- L.left.val = R.right.valL.left.val=R.right.val ：即 LL 的 左子节点 和 RR 的 右子节点 对称；
- L.right.val = R.left.valL.right.val=R.left.val ：即 LL 的 右子节点 和 RR 的 左子节点 对称。

根据以上规律，考虑从顶至底递归，判断每对节点是否对称，从而判断树是否为对称二叉树。

> 代码

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root == null) return true;
        return isSymmetric(root.left, root.right);
    }
    public boolean isSymmetric(TreeNode root1, TreeNode root2) {
        if(root1 == null && root2 == null) return true;
        if(root1 == null || root2 == null) return false;
        if(root1.val != root2.val) return false;
        return isSymmetric(root1.left, root2.right) && isSymmetric(root1.right, root2.left);
    }
}
```

# 27.顺时针打印矩阵-简单

> 题目

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

> 思路：主要是考虑边界值

> 代码

```java
class Solution {
    public int[] spiralOrder(int[][] matrix) {
        if(matrix.length == 0) return new int[0];
        int i = 0, j = 0, k = 0;
        int row = matrix.length, col = matrix[0].length;
        int[] res = new int[row * col];
        int index = 0;
        while(k < res.length){
            while(k < res.length && j < col - index) res[k++] = matrix[i][j++];
            j--;//由于执行完刚好到了边界外，所有需要回到边界处，后面一样的道理
            i++;//下一个开始的位置
            
            while(k < res.length && i < row - index) res[k++] = matrix[i++][j];
            i--;//回到边界处
            j--;//下一个开始的位置
            
            while(k < res.length && j >= index) res[k++] = matrix[i][j--];
            j++;//回到边界处
            i--;//下一个开始的位置
            
            while(k < res.length && i >= index + 1) res[k++] = matrix[i--][j];
            i++;//回到边界处
            j++;//下一个开始的位置
            
            index++;
        }
        return res;
    }
}
```

# 28、包含min函数的栈-简单

> 题目

设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

push(x) —— 将元素 x 推入栈中。
pop() —— 删除栈顶的元素。
top() —— 获取栈顶元素。
getMin() —— 检索栈中的最小元素。

> 思路:ListNode链表实现，每个结点都保存着这个结点至栈底的最小值，空间换时间

> 代码

```java
class MinStack {

    Node head;
    /** initialize your data structure here. */
    public MinStack() {
        
    }
    
    public void push(int x) {
        head = head == null ? new Node(x, x) : new Node(x, Math.min(head.min, x), head);
    }
    
    public void pop() {
       head = head.next;
    }
    
    public int top() {
        return head.val;
    }
    
    public int min() {
        return head.min;
    }
    private class Node {
        int val;
        int min;
        Node next;
        
        private Node(int val, int min) {
            this(val, min, null);
        }
        
        private Node(int val, int min, Node next) {
            this.val = val;
            this.min = min;
            this.next = next;
        }
    }
}
```

# 29.栈的压入、弹出序列-中等

> 题目

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

> 思路1：弹出序列的下一个元素必须在当前元素在压入序列的位置的” 前一个 “到最后这个范围

push：【1，2，3，4，5】，pop：【3， 4，2，5，1】

解释：第一个弹出元素为3，3在push中位置下标为2，所以下一个弹出元素必须为push中下标1到len-1这个范围的一个元素，只要不是这个范围的元素就无法弹出！但 “ 前一个 ”并不是简单的 在push中位置下标减一，需要排除掉已经弹出的元素，就是：

先弹出3，那么下一个弹出的只能是2、4、5其中一个（下标1至4），例子中弹出的是4（在push中下标为3），那么下一个弹出的只能是2、5其中一个（下标范围是1-4，而不是3-1=2至4，因为排除了3、4已经弹出了）

> 代码

```java
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Deque<Integer> stack = new ArrayDeque<>();
        for(int i = 0, j = 0; i < pushed.length; i++){
            stack.push(pushed[i]);
            //相等就一直弹出
            while(!stack.isEmpty() && stack.peek() == popped[j]){
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }
   
}
```

# 30. 从上到下打印二叉树-中等

> 题目

从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

> 思路：层序遍历

> 代码

```java
class Solution {
    public int[] levelOrder(TreeNode root) {
        if(root == null) return new int[0];
        Queue<TreeNode> queue = new ArrayDeque<>();
        List<Integer> list = new ArrayList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode temp = queue.poll();
            list.add(temp.val);
            if(temp.left != null) queue.offer(temp.left);
            if(temp.right != null) queue.offer(temp.right);
        }
        int[] res = new int[list.size()];
        int i = 0;
        for(int num: list){
            res[i++] = num;
        }
        return res;
    }
}
```

# 31. 从上到下打印二叉树2-中等

> 题目

从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。每一层区分开来

> 思路：层序遍历

每一次循环得到当前队列的size，这size个元素就是一层的元素

> 代码

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
       Queue<TreeNode> queue = new ArrayDeque<>();
       ArrayList<List<Integer>> res = new ArrayList<List<Integer>>();
       if(root == null) return res;
       queue.offer(root);
       while(!queue.isEmpty()){
           List<Integer> list = new ArrayList<>();
           //这里反过来可以少用一个中间变量存queue.size()
           for(int i = queue.size(); i > 0; i--){
               TreeNode temp = queue.poll();
               list.add(temp.val);
               if(temp.left != null) queue.offer(temp.left);
               if(temp.right != null) queue.offer(temp.right);
           }
           res.add(list);
       }
       return res;
    }
}
```

# 32. 从上到下打印二叉树3-中等

> 题目

从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。每一层元素顺序按做导游、右到左、左到右.....一直循环

> 思路：层序遍历

添加一个将list反过来就行

> 代码

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Deque<TreeNode> queue = new ArrayDeque<>();
        List<List<Integer>> lists = new ArrayList<>();
        if(root == null) return lists;
        int flag = 0;
        queue.offer(root);
        while(!queue.isEmpty()){
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            for(int i = 0; i < size; i++){
                TreeNode temp = queue.poll();
                list.add(temp.val);
                if(temp.left != null) queue.offer(temp.left);
                if(temp.right != null) queue.offer(temp.right);
            }
            if(flag == 1){
                Collections.reverse(list);
                flag = 0;
            }else{
                flag = 1;
            }
            lists.add(list);
        }
        return lists;
    }
}
```

# *33. 二叉搜索树的后序遍历序列-中等*

> 题目

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

> 思路1：递归判断每个结点是否其左子树结点都小于它，右子树结点都大于它

后序遍历最后一个结点是根节点，从前开始往后找到第一个大于根节点的值的位置，即根节点左右子树分界处，该位置前的结点应都小于根节点，后的结点则都必须大于根节点。继续递归判断左右子树每个结点即可。

> 代码

```java
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        return verifyPostorder(postorder, 0, postorder.length - 1);
    }
    public boolean verifyPostorder(int[] postorder, int beginIndex, int endIndex) {
        if(beginIndex >= endIndex) return true;
        int index = endIndex;
        //找到第一个大于根节点的值的位置，即根节点左右子树分界处，该位置前面的值应该都小于根节点，后面的应该都大于根节点
        for(int i = beginIndex; i <= endIndex; i++){
            if(postorder[i] > postorder[endIndex]){
                index = i;
                break;
            }
        }
        //前面肯定比根节点小，因为index为第一个大于根节点的结点位置
        // for(int i = beginIndex; i < index; i++){
        //     if(postorder[i] >= postorder[endIndex]){
        //         return false;
        //     }
        // }
        for(int i = index; i < endIndex; i++){
            if(postorder[i] <= postorder[endIndex]){
                return false;
            }
        }
        return verifyPostorder(postorder, beginIndex, index - 1) && verifyPostorder(postorder, index, endIndex - 1);
    }
}
```

# 34、二叉树中和为某一值的路径 -中等

> 题目

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

> 思路：DFS+剪枝

> 代码

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        pathSum(root, targetSum, new ArrayList<Integer>(), 0);
        return res;
    }
    public void pathSum(TreeNode root, int targetSum, List<Integer> list, int sum) {
        if(root == null){
            return;
        }
        list.add(root.val);
        sum += root.val;
        if(root.left == null && root.right == null){
            if(sum == targetSum){
                res.add(new ArrayList<Integer>(list));
            }
        }
        pathSum(root.left, targetSum, list, sum);
        pathSum(root.right, targetSum, list, sum);
        list.remove(list.size() - 1);
        sum -= root.val;
    }
}
```

总结

主要是使用DFS+剪枝：`遍历每一条路径判断其路径和同时栈帧弹出时对之前按加入路径的元素做一个删除，即剪枝`

# 35. 复杂链表的复制-中等

> 题目

请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

> 思路1：迭代+HashMap

先创建新结点只初始化值，并将老结点：新结点键值对加入map，然后再遍历老结点，新结点的next就是老结点的next对应的新结点，新结点的random就是老结点的random对应的新结点

> 代码

```java
public Node copyRandomList(Node head) {
        if(head == null) return null;
        //1、先复制正常的链表：val和next
        //2、同时把已有链表的每个结点与复制的节点一一对应放在Map里
        HashMap<Node, Node> map = new HashMap<>();
        Node tempHead = new Node(-1);
        Node newTemp = tempHead, oldTemp = head;
        while(oldTemp != null){
            //旧链表节点对应新链表的结点
            map.put(oldTemp,new Node(oldTemp.val));
            oldTemp = oldTemp.next;
        }
        //3、遍历复制了一半的链表，根据Map设置random指针
        newTemp = tempHead.next;
        oldTemp = head;
        while(newTemp != null){
            newTemp.random = map.get(oldTemp.random);
            newTemp = newTemp.next;
            oldTemp = oldTemp.next;
        }
        return tempHead.next;
    }
```

> 思路2：递归+HashMap

> 代码

```java
HashMap<Node, Node> map = new HashMap<>();
    public Node copyRandomList(Node head) {
        if(head == null) return null;
        //已经复制过了就直接返回对应已经复制的新结点
        if(map.containsKey(head)) return map.get(head);
        Node node = new Node(head.val);
        //已经被复制的结点将其与新结点的键值对保存起
        map.put(head, node);
        node.next = copyRandomList(head.next); 
        node.random = copyRandomList(head.random);
        return node;
    }
```

> 思路3：原地复制：将复制的新结点放在旧结点之后

旧链表：a->b->c->null

第一遍复制：a->A->b->B->c->C->null   (对应大写字母表示新复制的结点)

第二次遍历确定每一个新结点的random指针：新结点random指向旧结点的random指向的结点的下一个

第三次遍历将新旧结点分离。

> 代码

```java
public Node copyRandomList(Node head) {
        if(head == null) return null;
        Node temp1 = head;
        //在原链表的每个结点后赋值一个与自己相同的结点，但random指针还未赋值
        while(temp1 != null){
            Node node = new Node(temp1.val);
            node.next = temp1.next;
            temp1.next = node;
            temp1 = temp1.next.next;
        }
        //为新复制的结点的random指针复制
        temp1 = head;
        while(temp1 != null){
            //如果旧结点的random指向null就不做这操作，不然temp1.random.next会出错，而且新结点默认为null
            if(temp1.random != null){
                temp1.next.random = temp1.random.next;
            }
            temp1 = temp1.next.next;
        }
        //都搞定了就拆分两个链表，要保证两个链表的完整性才算复制成功！
        temp1 = head;
        Node temp2 = head.next;
        Node res = temp2;//将新结点头结点保存到待会返回
        while(temp2.next != null){
            temp1.next = temp1.next.next;
            temp1 = temp1.next;
            temp2.next = temp2.next.next;
            temp2 = temp2.next;
        }
        //旧结点最后一个指向null
        temp1.next = null;
        return res;
    }
```

总结

next与val容易复制，主要是考虑`如何根据旧结点的random设置新结点的random`。本题就使用了HashMap保存<旧, 新> 和 将新结点直接先复制到旧结点后面 两种方法实现了 “ 根据旧结点的random指向的结点设置新结点的random`”

# *36. 二叉搜索树与双向链表-中等*

> 题目

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

> 思路1：前后指针(栈实现)

定义一个结点记录中序遍历当前节点的前一个结点(开始时是null)，当前结点的左指针指向前面这个结点，前面这个结点的右指针指向当前结点。

递归、栈都可以，不过这样做完后最左边的结点的左指针和最右边的结点的右指针还是指向null的，需要单独处理一下。

> 代码

```java
class Solution {
    public Node treeToDoublyList(Node root) {
        if(root == null) return null;
        ArrayDeque<Node> stack = new ArrayDeque<>();
        Node prev = null, head = null;
        while(!stack.isEmpty() || root != null){
            while(root != null){
                stack.push(root);
                root = root.left;
            }
            Node cur = stack.pop();
            //当前结点左指针指向前面的结点
            cur.left = prev;
            //如果prev不为null，前面的结点的右指针指向当前结点
            if(prev == null){
                //prev为null时肯定是最开始遍历到最左边即最小的结点，将它保存起
                head = cur;
            }else{
                prev.right = cur;
            }
            //更新前结点
            prev = cur;
            root = cur.right;
        }
        //单独处理最右边结点的右指针和最左边结点的左指针,遍历完prev刚好就是指向最右边结点
        //最右边结点右指针指向头节点
        prev.right = head;
        //头节点左指针指向最右边结点
        head.left = prev;

        return head;
    }
}
```

> 思路2：前后指针(递归实现)

定义一个结点记录中序遍历当前节点的前一个结点(开始时是null)，`前结点的左指针指向前面这个结点，前面这个结点的右指针指向当前结点`。

递归、栈都可以，不过这样做完后最左边的结点的左指针和最右边的结点的右指针还是指向null的，需要单独处理一下。

> 代码

```java
class Solution {
    Node prev, head;
    public Node treeToDoublyList(Node root) {
        if(root == null) return null;
        treeToDoublyList1(root);
        //需要单独处理最左边的结点的左指针和最右边结点的右指针
        prev.right = head;//递归完prev就是最右边结点
        head.left = prev;
        return head;
    }
    public void treeToDoublyList1(Node root) {
        if(root == null) return ;
        treeToDoublyList1(root.left);
        //将每个结点左右指针指向前后结点，注意prev最开始为null，因此这样递归做完后
        //最左边的结点的左指针和最右边结点的右指针仍然指向null
        root.left = prev;
        if(prev != null){
            prev.right = root;
        }else{
            //此时第一次遍历到root.left==null这个结点，这结点就是最左边的
            head = root;
        }
        prev = root;
        treeToDoublyList1(root.right);
    }
}
```

> 总结

主要是：在中序遍历的基础上加上一个`前指针`，然后对前结点和当前结点做一个处理。

# 37 . 序列化二叉树-中等

> 题目

请实现两个函数，分别用来序列化和反序列化二叉树。

> 思路1：使用LinkedList实现的Queue接口的队列对象层序遍历

ArrayDeque对象当做队列使用将无法存储null值，LinkedList对象当做队列使用可以存放null值.

再根据层序遍历将其还原为二叉树。

> 代码

```java
public class Codec {

    // 使用LinkedList作为queue安装常规层序遍历操作得到层序遍历字符串
    public String serialize(TreeNode root) {
        if(root == null){
            return "";
        }
        StringBuilder res = new StringBuilder();
        res.append("[");
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            if(node != null){
                res.append("" + node.val);
                queue.offer(node.left);
                queue.offer(node.right);
            }else{
                res.append("null");
            }
            res.append(",");
        }
        res.append("]");
        // System.out.println(res);
        return res.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
       if(data == null || data.equals("")) return null;
       String[] list = data.substring(1, data.length() - 1).split(",");
       Queue<TreeNode> queue = new LinkedList<>();
       TreeNode root = new TreeNode(Integer.parseInt(list[0]));
       queue.offer(root);
       int i = 1;
       while(!queue.isEmpty() && i < list.length){
            TreeNode temp = queue.poll();
            if(!list[i].equals("null")){
                TreeNode node = new TreeNode(Integer.parseInt(list[i]));
                temp.left = node;
                queue.offer(node);
            }
            i++;
            if(!list[i].equals("null")){
                TreeNode node = new TreeNode(Integer.parseInt(list[i]));
                temp.right = node;
                queue.offer(node);
            }
            i++;
       }
        return root;
    }
    
}
```

> 总结

将二叉树转化为层序遍历的字符串形式（不需要严格按照层序遍历的满二叉树那种形式和完美的层序遍历形式（"[1,2,3,null,null,4,5]"））

将给的层序遍历列表转化为二叉树：出队，建立左右子结点并入队。

# 38. 字符串的排列-中等

> 题目

输入一个字符串，打印出该字符串中字符的所有排列。

 

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

> 思路：回溯

- 主要是需要考虑前面某个位置上的字符已经被选择过，那么那个位置上字符不能再被选择，这单可以利用一个set集合存放回溯过的字符下标，下次遍历回溯只能选择不再集合中位置上的字符，不过每次回溯完需要将前面加入set的下标删除，即“ 剪枝 ”。


- 不过还有一种情况，就是字符串里有重复字符，这时候会产生一样的排列，比如abcc，这是会有两个一样的下标排列0123和0132，他们都是abcc，所以这里需要再额外判断下。


> 代码

```java
class Solution {

    Set<String> list = new HashSet<>();
    public String[] permutation(String s) {
        if(s == null || s.equals("")) return new String[0];
        flashBack(s, "", new boolean[s.length()]);
        return list.toArray(new String[0]);
    }
    void flashBack(String s, String str, boolean[] visited){
        if(str.length() == s.length()){
            list.add(str);
            return ;
        }
        for(int i = 0; i <= s.length() - 1; i++){
            if(!visited[i]){940481
                visited[i] = true;
                flashBack(s ,s.charAt(i) + str ,visited);
                visited[i] = false;
            }
        }
    }
}
```

# 39 组中出现次数超过一半的数字-简单

> 题目

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

 

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

> 思路：摩尔投票

> 代码

```java
class Solution {
    public int majorityElement(int[] nums) {
        //遇到相同的数 计数器 加一，否则减一，如果为0则将结果数设置为当前数并将 计数器 赋值为1.
        int cnt = 0, res = 0;
        for(int num: nums){
            if(cnt == 0){
                res = num;
                cnt++;
            }else{
                if(res == num){
                    cnt++;
                }else{
                    cnt--;
                }
            }
        }
        return res;
    }
}
```

# 40. 最小的K个数-简单

> 题目

输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

> 思路1：堆

维护一个容量为K的大根堆，遍历数组先将大根堆填充到size为K，然后每加一个数都poll出去一个，这样最后数组所有元素都进入过一遍堆，最后剩下在堆里面的K个元素肯定是前K个最小的元素，堆顶则是第K小的元素。

> 代码

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if(arr.length == 0) return new int[0];
        if(k == arr.length) return arr;
        //默认为小根堆，需要添加比较器
        PriorityQueue<Integer> pQueue = new PriorityQueue<>(new Comparator<Integer>(){
            public int compare(Integer a, Integer b){
                return b - a;
            }
        });
        int[] res = new int[k];
        for(int i = 0; i < arr.length; i++){
            //前K个元素直接入堆，后面的元素只要比堆顶元素小就入堆
            pQueue.offer(arr[i]);
            if(i >= k) pQueue.poll();
        }
        //封装到数组中返回
        int i = 0;
        while(!pQueue.isEmpty()){
            res[i++] = pQueue.poll();
        }
        return res;
    }
}
```

> 思路2:快排

维护一个容量为K的大根堆，遍历数组先将大根堆填充到size为K，然后每加一个数都poll出去一个，这样最后数组所有元素都进入过一遍堆，最后剩下在堆里面的K个元素肯定是前K个最小的元素，堆顶则是第K小的元素。

> 代码

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if(arr.length == 0) return new int[0];
        quickSort(arr, 0, arr.length - 1, k);
        int[] res = new int[k];
        for(int i = 0; i < k; i++){
            res[i] = arr[i];
        }
        return res;
    }
    //递归进行快速排序，对获取到的基准值与K进行判断如何递归,最终将最小的K个数放在数组前K个位置
    public void quickSort(int[] arr, int beginIndex, int endIndex, int k){
        if(beginIndex >= endIndex) return ;
        int mid = getMidByMedian(arr, beginIndex, endIndex);
        if(mid + 1 == k) return ;
        if(mid + 1 < k) quickSort(arr, mid + 1, endIndex, k);
        else quickSort(arr, beginIndex, mid - 1, k);
    }

    //根据一个指定了区间的数组根据某个基准值进行划分,返回最后基准值的下标
    public int getMidIndex(int[] arr, int beginIndex, int endIndex){
        int index = beginIndex + 1;
        for(int i = beginIndex + 1; i <= endIndex; i++){
            if(arr[i] <= arr[beginIndex]){
                int temp = arr[i];
                arr[i] = arr[index];
                arr[index] = temp;
                index++;
            }
        }
        int temp = arr[beginIndex];
        arr[beginIndex] = arr[index - 1];
        arr[index - 1] = temp;
        return index - 1;
    }
    //随机选取基准值算法,将选好的基准值与区间第一个元素交换然后使用默认选取基准的方法进行划分
    public int getMidByRand(int[] arr, int beginIndex, int endIndex){
        Random random = new Random();
        int rand = random.nextInt(endIndex - beginIndex + 1);
        int index = rand + beginIndex;
        int temp = arr[beginIndex];
        arr[beginIndex] = arr[index];
        arr[index] = temp;
        return getMidIndex(arr, beginIndex, endIndex);
    }
    /**
     *取最左、中间、最右三个数的中位数作为基准进行划分，返回划分中间下标
     */
    public int getMidByMedian(int[] arr, int left, int right){
        int index = 0;
        int[] temp = new int[]{arr[left], arr[(left + right) / 2], arr[right]};
        Arrays.sort(temp);
        if(arr[left] == temp[1]){
            index = left;
        }
        if(arr[(left + right) / 2] == temp[1]){
            index = (left + right) / 2;
        }
        if(arr[right] == temp[1]){
            index = right;
        }
        int t = arr[left];
        arr[left] = arr[index];
        arr[index] = t;
        return getMidIndex(arr, left, right);
    }
}
```

# 41. 数据流中的中位数-简单

> 题目

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

> 思路：一个大根堆一个小根堆分别保存前一半小的元素和前一半大的元素。

> 代码

```java
class MedianFinder {

    PriorityQueue<Integer> bigPriority = new PriorityQueue<>((a, b)->{
        return b - a;
    });
    PriorityQueue<Integer> smallPriority = new PriorityQueue<>();
    /** initialize your data structure here. */
    public MedianFinder() {
        
    }
    
    public void addNum(int num) {
        //先添加到大顶堆，再添加小顶堆，一直循环添加，保证大顶堆与小顶堆元素个占一半，且前者元素小于等于后者。
        if(bigPriority.size() == 0){
            bigPriority.offer(num);
            return;
        }
        if(bigPriority.size() == smallPriority.size()){
            if(num <= smallPriority.peek()){
                bigPriority.offer(num);
            }else{
                bigPriority.offer(smallPriority.poll());
                smallPriority.offer(num);
            }
        }else{
            if(num >= bigPriority.peek()){
                smallPriority.offer(num);
            }else{
                smallPriority.offer(bigPriority.poll());
                bigPriority.offer(num);
            }
        }
    }
    
    public double findMedian() {
        //如果是偶数 两个堆的size肯定不等，此时返回两堆顶元素除以2，否则返回大顶堆的堆顶元素
        if(bigPriority.size() == smallPriority.size()){
            return (bigPriority.peek() + smallPriority.peek()) * 1.0 / 2;
        }else{ 
            return (double)bigPriority.peek();
        }
    }
}
```

# 42. 连续子数组的最大和-简单

> 题目

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

> 思路：动态规划

状态：dp[i]表示*以下标i这个位置元素结尾*的子数组的最大和。

转移方程：如果dp[i] > 0，那么dp[i+1] = dp[i] + nums[i+1]，否则dp[i+1] = nums[i+1]

> 代码

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int res = nums[0];
        for(int i = 1; i < nums.length; i++){
            if(nums[i - 1] > 0) nums[i] += nums[i - 1];
            if(nums[i] > res) res = nums[i];
        }
        return res;
    }
}
```

# 43.  1～n 整数中 1 出现的次数-困难

> 题目

输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。

例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

> 思路：递归

[大佬题解](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/solution/javadi-gui-by-xujunyi/)

> 代码

# 44. 数字序列中某一位的数字

# 45. 把数组排成最小的数-中等

> 题目

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

> 思路：排序

大小比较规则：num1+num2 < num2 + num1

> 代码

```java
class Solution {
    public String minNumber(int[] nums) {
        if(nums.length == 0) return "";
        //使用比较器需要将基本类型数组转化位对应的包裹类型
        Integer[] temp = new Integer[nums.length];
        for(int i = 0; i < nums.length; i++){
            temp[i] = nums[i];
        }

        Arrays.sort(temp, (a, b)->{
            return (a + "" + b).compareTo(b + "" + a);
        });

        StringBuffer res = new StringBuffer();
        for(int num: temp){
            res.append(num);
        }
        return res.toString();
    }
}
```

# 46.  把数字翻译成字符串-中等

> 题目

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。



> 思路：动态规划

状态：dp[i]表示前i个数字能表示的字符串个数

转换方程：如果第i个数字能和前面的数字组合表示一个字符：dp[i] = dp[i-1]+dp[i-2]；否则dp[i] = dp[i-1];

初始化：dp[0] = dp[1] = 1;

> 代码

```java
class Solution {
    public int translateNum(int num) {
        //第i位结尾的数字如果能和第i-1位的数组合翻译为一个字符（0-25），则其翻译成字符串的个数 = 第i - 2位的 + 第i- 1位的
        //否则就等于第i- 1位的
        String str = num + "";
        int[] res = new int[str.length()];
        res[0] = 1;
        for(int i = 1; i < str.length(); i++){
            int temp = Integer.parseInt(str.charAt(i - 1) + "" + str.charAt(i));
            if(temp > 25 || str.charAt(i - 1) == '0'){
                res[i] = res[i - 1];
            }else{
                res[i] = res[i - 1] + (i - 2 >= 0 ? res[i - 2] : 1);
            }
        }
        return res[str.length() - 1];
    }
}
```

# 47. 礼物的最大价值-中等

> 题目

在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

题目解析：从最左上角到最右下角；只能向右和向下走；

> 思路1：动态规划(自顶向下)

往右边走、与往下边走两个选择，选最大的即可。考虑到一些数据重复递归计算耗费大量时间，可以定义个状态数组存储已经计算过的到达该位置能得到的最大权重

> 代码

```java
class Solution {
    int[][] dp;
    public int maxValue(int[][] grid) {
        dp = new int[grid.length][grid[0].length];
        return maxValue(grid, 0, 0);
    }
    public int maxValue(int[][] grid, int i, int j) {
        if(i >= grid.length || j >= grid[0].length) return 0;
        if(dp[i][j] != 0) return dp[i][j];
        int cur = grid[i][j];
        dp[i][j] = Math.max(maxValue(grid, i + 1, j) + cur, maxValue(grid, i, j + 1) + cur);
        return dp[i][j];
    }

}
```

> 思路1：动态规划(自底向上)



> 代码

```java
class Solution {
    public int maxValue(int[][] grid) {
        int row = grid.length, col = grid[0].length;
        int[][] dp = new int[row][col];
        //dp[i][j]表示到从这个位置开始到达最右下角的最大价值
        //初始化：dp[i][j] = grid[i][j]
        //如果是最右边一列和最下面一列：dp[i][j] = dp[i + 1][j] + dp[i][j]或dp[i][j - 1] + dp[i][j]
        //其它列：dp[i][j] = max(dp[i + 1][j],dp[i][j + 1])
        for(int i = row - 1; i >= 0; i--){
            for(int j = col - 1; j >= 0; j--){
                if(i == row - 1 && j == col - 1) dp[i][j] = grid[i][j];
                else if(i == row - 1) dp[i][j] = grid[i][j] + dp[i][j + 1];
                else if(j == col - 1) dp[i][j] = grid[i][j] + dp[i + 1][j];
                else dp[i][j] = grid[i][j] + Math.max(dp[i + 1][j], dp[i][j + 1]);
            }
        }
        return dp[0][0];
    }

}
```

# 48.最长不含重复字符的子字符串-中等

> 题目

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度？

> 思路1：滑动窗口

前后指针left与right，没有遇到重复的话left保持不动right一直向前走，他们两个之间的距离就是当前无重复子字符串的长度；一旦遇到重复的，left指针就向右边移动到重复字符位置的下一个字符位置，然后继续保持不动right继续向前走直到遇到下一个重复的。不停循环到right到字符串长度，循环过程中更新maxLen即可。

那怎样确定是否出现重复字符并且确定重复字符在字符串中的位置呢？可以使用各种各样的方式实现：

1. 队列：遇到重复的就将元素出队到不再重复，此时队头就刚好是重复字符的下一个字符。
2. 常规HashMap：key为字符，value为字符位置
3. 数组实现HashMap：由于是ASCLL字符，所有可以定义个数组实现HashMap，下标为字符，值为字符位置。

> 代码

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Queue<Character> queue = new ArrayDeque<>();
        int maxLen = 0;
        for(int i = 0; i < s.length(); i++){
            while(queue.contains(s.charAt(i))){
                queue.poll();
            }
            queue.offer(s.charAt(i));
            maxLen = Math.max(queue.size(), maxLen);
        }
        return maxLen;
    }
}
```

> 代码2：使用数组存放字符下标，这样`既可以判断是否重复还可以直接得到重复字符的位置`

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int maxLen = 0;
        int index[] = new int[300];//自动初始化为0
        for(int i = 0, j = 0; i < s.length(); i++)
        {
            //取两者较大的值非常巧妙：如果说明当前子串范围内出现了重复字符则index[s.charAt(i)]一定比j大
            //且存放的重复字符位置的下一个。如果index[s.charAt(i)]比j小则表示当前子串范围外有字符重复，不用管。
            j = Math.max(index[s.charAt(i)], j);
            
            //更新maxLen值
            maxLen = Math.max(i - j + 1, maxLen);
            //将字符保存到数组中
            index[s.charAt(i)] = i + 1;
        }
        return maxLen;
    }
}
```

# 49.丑数-中等

> 题目

我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

> 思路：没得

8=2*2\*2;质因子为2；12=2\*2\*3质因子为2和3，14=2\*7质因子为2和7 ........ 。把质因子只包含2、3、5三个数一个或多个的数称为丑数，规定1为丑数。丑数乘以2 、3 、5肯定就是丑数，所以我们只需从1这个给丑数开始不停乘以2、3、5，产生的新的丑数又可以乘以2、3、5，这样就不停的诞生丑数，要按顺序求第n个就需要注意下乘的顺序，每次诞生的丑数必须是丑数与2、3、5乘积的最小的一个。

1 x 2  1 x 3  1 x 5 ; 

1 x 3  1 x 5  2 x 3;

2 x 3  1 x 5  2 x 3;

..........

> 代码:

```java
class Solution {
    public int nthUglyNumber(int n) {
        int[] res = new int[n];
        res[0] = 1;
        int i = 0, j = 0, k = 0;
        for(int x = 1; x < res.length; x++){
            res[x] = Math.min(res[i] * 2, Math.min(res[j] * 3,res[k] * 5));
            if(res[x] == res[i] * 2) i++;
            if(res[x] == res[j] * 3) j++;
            if(res[x] == res[k] * 5) k++;
        }
        return res[n - 1];
    }
}
```

# 50.第一个只出现依次的字符-简单

> 题目

在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

> 思路：LinkedHashMap

key为字符，value为该字符是否重复

> 代码:

```java
class Solution {
    public char firstUniqChar(String s) {
       //维护一个hashmap，key存字符，value存是否重复
       Map<Character, Boolean> map = new LinkedHashMap<>();
       //遍历字符串更新map
       char[] str = s.toCharArray();
        for(char ch: str){
            map.put(ch, !map.containsKey(ch));
        }
       //遍历map找到第一个没重复的
       for(char ch: map.keySet()){
           if(map.get(ch)) return ch;
       }
       return ' ';
    }
}
```

# 51.数组中的逆序对-困难

> 题目

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。。

> 思路：归并排序

数组左边区域与右边区域合并的过程中如果左边某个数大于右边的某个数，那么此时左边的这个数和该数后面的所有数(在左边区域内)都可以与右边这个数构成逆序对，所有此时执行sum += mid - i + 1;

> 代码:

```java
class Solution {
    //常规比较需要n^2的时间复杂度，利用归并排序的比较过程可以实现计数，时间复杂度变为nlogn
    int sum = 0;
    public int reversePairs(int[] nums) {
        mergeSort(nums, 0, nums.length - 1);
        return sum;
    }
    //归并排序
    public void mergeSort(int[] arr, int left, int right){
        if(left >= right){
            return ;
        }else{
            int mid = left + (right - left) / 2;
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            if(arr[mid] > arr[mid + 1])
            merge(arr, left, mid, right);
        }
    }
    //区间归并操作，在归并比较大小的同时计数逆序对个数
    public void merge(int[] arr, int left, int mid, int right){
        int[] temp = new int[right - left + 1];
        int i = left, j = mid + 1, k = 0;
        while(i <= mid && j <= right){
            if(arr[i] <= arr[j]){
                temp[k] = arr[i++];
            }else{
                sum = sum + mid - i + 1;
                temp[k] = arr[j++];
            }
            k++;
        }
        while(i <= mid){
            temp[k++] = arr[i++];
        }
        while(j <= right){
            temp[k++] = arr[j++];
        }
        k = 0;
        for(int x = left; x <= right; x++){
            arr[x] = temp[k++];
        }
    }
}
```

# 52、 两个链表的第一个公共节点

> 题目

输入两个链表，找出它们的第一个公共节点。

> 思路：双指针

注意：不相交的两条链表可以看作在最后的`null`相交。所以要把`null`看作是最后一个节点，如果使用`结点.next`是否为null来判断是否到达链表尾的话就没有把`null`当成最后一个结点了，这样的话对于不相交的两个链表就需要单独额外考虑了，也行但不够方便。

> 代码:

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        //两个指针初始化都指向两条链表开始结点，不相同就都往后走
        //如果走到最后一个结点（最后一个结点是null）从另外一条链表的头节点开始继续走（这就相当于弥补了两条链表非公共区间之间的差值）
        //第二次两个指针在两天链表上交换了位置走一定就会同步走到相互交互的结点上。
        ListNode point1 = headA, point2 = headB;
        while(point1 != point2){
            if(point1 == null){
                point1 = headB;
            }else{
                point1 = point1.next;
            }
            if(point2 == null){
                point2 = headA;
            }else{
                point2 = point2.next;
            }
        }
        return point1;
    }
}
```

# 53.在排序数组中查找数字 I-简单

> 题目

统计一个数字在排序数组中出现的次数。

> 思路1：二分

找到目标值左右两边的边界下标

细节1：循环终止条件是否包含等于

细节2：等于目标值的时候left与right是变为mid还是变为mid + 或 - 1

`循环终止条件包含等于更好，这样后续与目标值相等时就直接使用mid +或- 1，这样最后遍历完肯定是大于或小于目标值的第一个数。`

> 代码:

```java
class Solution {
    public int search(int[] nums, int target) {
        if(nums.length == 0) return 0;
        //二分法:找到第一个小于等于目标数的值的位置和第一个大于目标数的值的位置
        //找第一个小于等于：等于的时候往左边分
        //找第一个大于：等于的时候往右边分
        int left = 0, right = nums.length - 1;
        int mid = 0;
        while(left < right){
            mid = left + (right - left) / 2;//向下取整
            if(nums[mid] >= target){
                right = mid;
            }else{
                left = mid + 1;
            }
        }
        if(nums[left] != target){
            return 0;
        }
        int begin = left;
        right = nums.length - 1;
        while(left <= right){
            mid = left + (right - left) / 2;//向下取整
            if(nums[mid] > target){
                right = mid - 1;
            }else{
                left = mid + 1;
            }
        }
        return left - begin;
    }
}
```

> 思路2：优化，

`循环终止条件包含等于更好，这样后续与目标值相等时就直接使用mid +或- 1，这样最后遍历完肯定是大于或小于目标值的第一个数。`

独立一个方法出来，该方法是`返回大于给定参数的第一个数在数组中的位置`

> 代码

```java
class Solution {
    public int search(int[] nums, int target) {
        //target不存在nums中时返回的要么都是0，要么都是nums.length
        return helper(nums, target) - helper(nums, target - 1);
    }
    //如果tar不存在于数组中那么返回的就是0（i不会移动）或nums.length（i移动到数组尾）
    int helper(int[] nums, int tar) {
        int i = 0, j = nums.length - 1;
        while(i <= j) {
            int m = (i + j) / 2;
            if(nums[m] <= tar) i = m + 1;
            else j = m - 1;
        }
        return i;
    }
}
```

# 54. 0～n-1中缺失的数字-简单

> 题目

一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

> 思路：二分

长度为n的数组升序存储[0, n]范围的数字，相当于是一个值与下标完全一样的长为n+1的数组从中拿走了一个值，拿走的这个值的位置前面的所有数组值与下标仍然一样，只是后面的数组值与下标不一样（大1），所以我们可以通过二分去`找到第一个值与下标不一样的元素位置`或`最后一个值与下标一样的元素位置。`

> 代码:

```java
class Solution {
    public int missingNumber(int[] nums) {
        //二分：被拿出去的值的位置的左边的元素的值与自己下标一致，右边元素的值与其下标就不一致
        int left = 0, right = nums.length - 1;
        int mid = 0;
        while(left <= right){
            mid = left + (right - left) / 2;
            if(nums[mid] == mid){
                left = mid + 1;
            }else{
                right = mid - 1;
            }
        }
        //left刚好就是第一个元素与下标不相等的位置，如果都相等，left也刚好等于nums.length
        return left;
    }
}
```



# 二分细节总结

1、left必须变为mid + 1，不能变为mid，因为mid是向下取整计算到的，如果这样做可能会陷入死循环，用`while（left < right）`这个也不行，除非额外增加其它判断来防止。

2、循环终止条件如果不包含左右指针相等即`while（left < right）`时，需要额外考虑left==right时的情况。

> 使用总结

当我们需要左右指针最终停留位置时用`while（left < right）`要好一些，因为最终left与right肯定是相等的，最后停的位置取left和right都一样，但此时要注意考虑特殊情况。



# 55.二叉搜索树的第k大节点-简单

> 题目

给定一棵二叉搜索树，请找出其中第k大的节点。

> 思路：栈或递归实现反过来的中序遍历

> 代码:

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {

    public int kthLargest(TreeNode root, int k) {
        //使用栈来 伪后序遍历
        Deque<TreeNode> stack = new ArrayDeque<>();
        //遍历时维护一个计数器，等于K时就返回其值。
        int cnt = 1;
        while(!stack.isEmpty() || root != null){
            while(root != null){
                stack.push(root);
                root = root.right;
            }
            TreeNode top = stack.pop();
            if(cnt == k){
                return top.val;
            }else{
                cnt++;
                root = top.left;
            }
        }
        return -1;
    }
}
```

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    int cnt = 1, res;
    public int kthLargest(TreeNode root, int k) {
        largest(root, k);
        return res;
    }
    public void largest(TreeNode root, int k){
        //cnt大于k时表示已经得到了第k大数字，此时没必要再继续递归下去了。
        if(root == null || cnt > k) return ;
        largest(root.right, k);
        if(cnt == k){
            res = root.val;
        }
        cnt++;
        largest(root.left, k);
    }
}
```



# 56.二叉树的深度-简单

> 题目

输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

> 思路1：递归

> 代码:

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if(root == null) return 0;
        int leftDepth = maxDepth(root.left);
        int rightDepth = maxDepth(root.right);
        return Math.max(leftDepth, rightDepth) + 1;
    }
}
```

> 思路2：层序遍历

> 代码:

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int maxDepth(TreeNode root) {
        //使用队列层序遍历计算深度
        Queue<TreeNode> queue = new ArrayDeque<>();
        if(root == null) return 0;
        int cnt = 0;
        queue.offer(root);
        while(!queue.isEmpty()){
            cnt++;
            for(int i = queue.size(); i > 0; i--){
                TreeNode node = queue.poll();
                if(node.left != null) queue.offer(node.left);
                if(node.right != null) queue.offer(node.right);
            }
        }
        return cnt;
    }
}
```

# 57.数组中数字出现的次数-中等

> 题目

一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

> 思路：主要是异或运算

异或运算性质：与0进行异或等于本身，相等的两个数异或等于0；

运用交换律

> 代码

```java
class Solution {
    public int[] singleNumbers(int[] nums) {
        if(nums.length == 0) return new int[0];
        //得到两个没有重复的树的异或运算结果
        int res1 = 0;
        for(int num: nums){
            res1 ^= num;
        }
        //在异或运算结果数二进制表示中随便确定一个等于1的某二进制位，制造一个该位是1其余位都为0的数。
        int cnt = 0;
        while((res1 & 1) != 1){
            res1 >>>= 1;
            cnt++;
        }
        int flag = 1 << cnt;
         //等于1的某二进制位：表示这两个数在这一位肯定不同，一个数在这位是1，另一个数在这一位是0，我们
        //就可以以此来区分出这两个数，再来两遍异或运算把某个数剔除不参与异或运算，最终结果就是另外一个数了。
        int[] res = new int[2];
        for(int num: nums){
            if((num & flag) == 0){
                res[0] ^= num;
            }else{
                res[1] ^= num;
            }
        }
        return res;
    }
}
```

# 58.数组中数字出现的次数2-中等

> 题目

在一个数组 `nums` 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

> 思路1：普通位运算

如果一个数字出现3次，它的二进制每一位也出现的3次。如果把所有的出现三次的数字的二进制表示的每一位都分别加起来，那么每一位都能被3整除。 我们把数组中所有的数字的二进制表示的每一位都加起来。如果某一位能被3整除，那么只出现一次的那个数在这一位肯定为0，因为如果它在这一位是1了，那么所有数在这一位的和始终比3的倍数多出一位（因为其它数都是重复三次出现）。

> 代码

```java
class Solution {
    public int singleNumber(int[] nums) {
        StringBuffer str = new StringBuffer();
        int temp = 0;
        int flag = 1 << 31;
        for(int i = 0; i < 32; i++){
            temp = 0;
            //计算所有数这一位的和
            for(int j = 0; j < nums.length; j++){
                if((nums[j] & flag) != 0){
                    temp++;
                }
            }
            //由 和 来判断特殊的那一个数在这一位是0还是1         
            str.append(temp % 3);
            flag >>>= 1;
        }
        return Integer.parseInt(str.toString(), 2);
    }
}
```

> 思路2：状态机

一个二进制位x与另一个二进制位y进行异或运算：如果y是0结果就等于x本身.，y与x相同结果就变为0，相当于:0->1->0

我们自己也可以实现一种运算规则：当然y是0结果就等于x本身这个规则不变；y与x相同仍然变为0（但此时用另一个变量记录着已经遇到过一次了），再遇到一个z还是相同x还是0不变（此时重置记录变量）；相当于：x为0->1->0->0，记录变量位0->0->1->0

> 代码

```java
class Solution {
    public int singleNumber(int[] nums) {
       int flag = 0;//标记位
       int res = 0;//结果位：第三次遇到相同才变为0
       for(int num: nums){
           /*
           如果标记位是0那么结果位和常规一样做异或运算
           if(flag == 0){
               res ^= num;
           }如果标记位是1，即这个位已经出现了两次相同的数了，第三次再出现结果位也得是0
           else{
               res = 0;
           }
           结果位是0只有两种情况，一种是本身是0然后num也位0，另一种是本身是1同时num也为1
           所以flag是否标记为1就看是不是第二种情况，是就标记为1，不是就不标记即为0，刚好满足下面的表达式
           if(res == 0){
               flag = flag ^ num;
           }结果位是1标志位肯定没有标记
           else{
               flag = 0;
           }
           */
           res = (res ^ num) & (~flag);
           flag = (flag ^ num) & (~res);
       }
       return res;
    }
}
```

# 59.和为s的两个数字-简单

> 题目

输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

> 思路：双指针

> 代码

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while(left < right){
            if(nums[left] == target - nums[right]){
                return new int[]{nums[left], nums[right]};
            }else if(nums[left] > target - nums[right]){
                right--;
            }else{
                left++;
            }
        }
        return new int[0];
    }
}
```

# 60.和为s的连续正数序列-简单

> 题目

输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

> 思路：双指针：滑动窗口

> 代码

```java
class Solution {
    public int[][] findContinuousSequence(int target) {
        int left = 1, right = 2, sum = 3;
        if(target < 3) return new int[0][0];
        ArrayList<int[]> res = new ArrayList<>();
        while(left <= target / 2){
            if(sum == target){
                int[] temp = new int[right - left + 1];
                for(int i = left, j = 0; i <= right; i++, j++){
                    temp[j] = i;
                }
                res.add(temp);
                //注意这里也需要对指针进行移动，移动left或right都行。
                right++;
                sum += right;
            }else if(sum < target){
                right++;
                sum += right;
            }else{
                sum -= left;
                left++;
            }
        }
        return res.toArray(new int[0][0]);
    }
}
```

# 61.翻转单词顺序-简单

> 题目

输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

> 思路：双指针

trim() split() substring()

> 代码

```java
class Solution {
    public String reverseWords(String s) {
        s = s.trim();
        if(s == null || s.length() == 0) return "";
        char[] str = s.toCharArray();
        int left = str.length - 1, right = str.length - 1;
        StringBuffer res = new StringBuffer();
        while(left >= 0){
            while(left >= 0 && str[left] != ' ') left--;
            res.append(s.substring(left + 1, right + 1));
            res.append(" ");
            while(left >= 0 && str[left] == ' ') left--;
            right = left;
        }
        return res.toString().substring(0, res.length() - 1);
    }
}
```

# 62. 左旋转字符串-简单

> 题目

字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

> 思路：使用substring()或自己字符拼接

> 代码

```java
class Solution {
    public String reverseLeftWords(String s, int n) {
        n %= s.length();
        if(n == 0) return s;
        return s.substring(n, s.length()) + s.substring(0, n);
    }
}
```

# 63. 滑动窗口的最大值-困难

> 题目

给定一个数组 `nums` 和滑动窗口的大小 `k`，请找出所有滑动窗口里的最大值。

> 思路1:单调队列

最大元素的位置的前面的元素被移出去时不会影响最大值，所以可以使用单调递减队列保存当前窗口的数，队列尾元素如果小于即将入队的元素就pollLast出去，知道队列尾元素大于等于即将入队的元素。这样队头就是当前窗口的最大值。

> 代码

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums.length == 0) return new int[0];
        Deque<Integer> deque = new ArrayDeque<>();
        int[] res = new int[nums.length - k + 1];
        for(int i = 0; i < k; i++){
            while(!deque.isEmpty() && deque.peekLast() < nums[i]){
                deque.pollLast();
            }
            deque.offerLast(nums[i]);
        }
        res[0] = deque.peekFirst();
        int i = 1, j = k, index = 1;
        while(j < nums.length){
            if(nums[i - 1] == deque.peekFirst()){
                deque.pollFirst();
            }
            while(!deque.isEmpty() && deque.peekLast() < nums[j]){
                deque.pollLast();
            }
            deque.offerLast(nums[j]);
            res[index++] = deque.peekFirst();
            i++;
            j++;
        }
        return res;
    }
}
```

> 思路2：暴力

> 代码

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int len = nums.length;
        int[] res = new int[len - k + 1];
        int preMaxIndex = max(nums, 0, k - 1);
        res[0] = nums[preMaxIndex];
        for(int i = 1; i < res.length; i++){
            //如果之前窗口最大值下标刚好是第一个那么下一个窗口最大值就重新计算，否则就可以
            //直接拿之前窗口最大值和新增元素作比较
            if(preMaxIndex == i - 1){
                preMaxIndex = max(nums, i, i + k - 1);
            }else if(nums[preMaxIndex] <= nums[i + k - 1]){
                preMaxIndex = i + k - 1;
            }
            res[i] = nums[preMaxIndex];
        }
        return res;
    }
    public int max(int[] nums, int begin, int end) {
        int maxIndex = begin;
        for(int i = begin; i <= end; i++){
            if(nums[maxIndex] <= nums[i])
                maxIndex = i;
        }
        return maxIndex;
    }
}
```

# 64. n个骰子的点数-中等

> 题目

把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率

> 思路1：动态规划（自顶向下）

状态：n枚骰子点数和为X的可能情况等于`n - 1枚骰子点数和为 X - 1 、X - 2、 X -3 ... X - 6的所有可能情况相加`

> 代码

```java
class Solution {
    int[][] dp;
    public double[] dicesProbability(int n) {
        dp = new int[n + 1][6 * n + 1];
        int sum = (int)Math.pow(6, n);
        double[] res = new double[5 * n + 1];
        for(int i = n; i <= 6 * n; i++){
            res[i - n] = dicesProbability(n, i) * 1.0 / sum;
        }
        return res;
    }
    //n个骰子和为X的次数
    public int dicesProbability(int n, int x){
        if(x < n || x > 6 * n) return 0;
        if(dp[n][x] != 0) return dp[n][x];
        if(n == 1) {
            dp[n][x] = 1;
            return 1;
        }

        int res = 0;
        for(int i = 1; i <= 6; i++){
            res += dicesProbability(n - 1, x - i);
        }
        dp[n][x] = res;
        return res;
    }
}
```

> 思路：动态规划（自底向上）

`状态`：dp[i\][j]表示i枚骰子总共点数和为j的可能有多少种。

`转移方程`：dp[i\][j] = dp[i - 1\][j - k]。其中k为[1, 6]。含义是拿一个骰子出来，它出现的数字可能性（一个骰子只有一种可能）加上剩余i-1枚骰子数字的和为：减去拿出来的一枚骰子的数的可能性

`空间优化`：由于只用到了上一个阶段的数据（n-1枚骰子）所以可以不用把全部的数据都记录着

**空间优化时注意点：**

1. 需要额外一个数组保留上一阶段的值：因为如果只使用一个数组的话填充新的值时会覆盖上个阶段的值，后面再用到上个阶段的值时就用成了这个阶段新填充的值。
2. 新填充时需要初始化为0：因为保存着上一个阶段的值
3. j - k必须在上一轮合法范围内才能算



如果这一阶段`从后面开始填充`就不会覆盖那就不需要额外的一个数组

> 代码

```java
class Solution {
    public double[] dicesProbability(int n) {
        int[] before = new int[6 * n + 1];
        int[] after = new int[6 * n + 1];
        for(int i = 1; i <= 6; i++){
            before[i] = 1;
        }
        for(int i = 2; i <= n; i++){
            for(int j = i; j <= i * 6; j++){
                //细节1：必须初始化为0
                after[j] = 0;
                for(int k = 1; k <= 6; k++){
                    //细节2：j - k必须在上一轮合法范围内才能算
                    if(j - k > 0 && j - k <= (i - 1) * 6 && j - k >= i - 1) 
                        after[j] += before[j - k];
                }
            }
            int[] temp = before;
            before = after;
            after = temp;
        }
        double[] res = new double[5 * n + 1];
        int sum = (int)Math.pow(6, n);
        for(int i = n; i <= 6 * n; i++){
            res[i - n] = before[i] * 1.0 / sum;
        }
        return res;
    }
}
```

# 65. 扑克牌中的顺子-简单

> 题目

从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

> 思路1：常规思路

1、不能有重复的

2、排好序，遍历时得到0的个数，当两数的差不为1时用0去填补，如果0填补完了还不够返回false

> 代码

```java
class Solution {
    public boolean isStraight(int[] nums) {
        int flag = 0, zeroCount = 0;
        Arrays.sort(nums);
        for(int i = 0; i < nums.length - 1; i++){
            //计算0的个数
            if(nums[i] == 0){
                zeroCount++;
                continue;
            }
            //计算两个数的差值，如果为0表示相等之间返回false
            flag = nums[i + 1] - nums[i];
            if(flag == 0) return false;
            //如果两数相差大于1就会消耗0的个数，zereCount小于0表示不够返回false
            zeroCount = zeroCount - flag + 1;
            if(zeroCount < 0) return false;
        }
        return true;
    }
}
```

> 思路2：优化第二个条件

1、不能有重复的

2、正常情况对于五张牌如果是顺子那么五张牌里的最大值与最小值`一定等于4`；但如果有赖子牌（大小王）可以替代任何牌那么就不一定了：

- 如果赖子`替代的是顺子中间部分的牌`（1、赖子、赖子、4、5）那么最大值与最小值的差值不变还是4。
- 如果`替代的牌有顺子的头或尾的牌`那么最大值与最小值的差值显然就会小于4（赖子、2、赖子、4、5）。

综上分析满足顺子的第二个条件为：`max - min < 4`

> 代码

```java
class Solution {
    public boolean isStraight(int[] nums) {
        int min = 15, max = -1;
        Arrays.sort(nums);
        for(int i = 0; i < nums.length; i++){
            if(nums[i] == 0){
                continue;
            }
            if(i > 0 && nums[i] == nums[i - 1]){
                return false;
            }
            if(min > nums[i]) min = nums[i];
            if(max < nums[i]) max = nums[i];
        }
        return max - min < 5;
    }
}
```

> 思路3：优化第一个条件

有了第二个条件的优化后我们想要判断是否有重复牌其实可以不使用排序了，我们可以用这几种来判断

- 使用HashMap判断是否有重复牌

- 由于数字范围有限（0-13），可以使用boolean数组来代替HashMap

> 代码

略

# 66.  圆圈中最后剩下的数字(约瑟夫环)-中等

> 题目

0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

> 思路1：常规思路：使用List来循环删除

`删除元素的下标 =  （ 开始下标 + 间隔m - 1 ） % 当前列表长度`

开始下标开始是0，每次循环删除后更新开始下标即可，开始下标更新分析如下：

- 除了`删除元素的下标`刚好为`列表最后一位`那么下次的开始下标就是数组开始，其它情况下次的开始下标和删除下标是一致的。

所有：下一次的开始下标 = 本次删除下标 % 删除后的列表长度（即原先长度 - 1）

```java
class Solution {
    public int lastRemaining(int n, int m) {
        ArrayList<Integer> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            list.add(i);
        }
        //开始下标开始是0
        int beginIndex = 0;
        while (list.size() > 1) {
            //本次移除元素的下标等于开始下标加上间隔m减一然后对长度取余
            int removeIndex = (beginIndex + m - 1) % list.size();
            list.remove(removeIndex);
            //更新开始下标，如果删除下标刚好为列表最后一位那么下次的开始下标就是数组开始，其它情况其实
            //下次的开始下标和删除下标一致的。
            beginIndex = removeIndex % list.size();
        }
        return list.get(0);
    }
}
```

> 思路2：动态规划（自顶向下）

[大佬题解](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/solution/jian-zhi-offer-62-yuan-quan-zhong-zui-ho-dcow/)



`状态`：f(n, m)表示[0, n-1]，间隔为m，从第一个数开始，返回最后剩下的数。

主要是分析·`f(n, m)`怎么根据子问题`f(n-1, m)`的结果得到答案。例如n=5，m=3：

`f(n, m) = 求0,1,2,3,4从第一个数开始，间隔为3，最后剩下的一个数`，我们看看它执行了一轮后也就是删除了第一个该删除的元素，这里是删除2，下一次该从3开始，相当于：求3,4,0,1从第一个数开始，间隔为3，最后剩下的一个数,即：

`求0,1,2,3,4从第一个数开始，间隔为3，最后剩下的一个数 ` 等价于 `求3,4,0,1从第一个数开始，间隔为3，最后剩下的一个数`

再看`f(n-1, m) = 求0,1,2,3从第一个数开始，间隔为3，最后剩下的一个数 。`我们对比下f(n, m)：

​		`f(n, m) = 求3,4,0,1从第一个数开始，间隔为3，最后剩下的一个数`



我们可以感觉出它们两者的每个数之间是存在什么关系的。通过观察+运气+天才找规律得出f(n-1, m)的数与f(n, m)的数存在对应关系：f(n, m) = (f(n-1, m) + m % n) % n = (f(n-1, m) + m) % n。

> 代码

```java
public int lastRemaining2(int n, int m) {
    	//边界：约瑟夫环长度为1，剩下的就是下标0
        if(n == 1) return 0;
        int res= lastRemaining(n - 1, m);
        return (res + m) % n;
    }
```

> 思路3：动态规划（自底向上）

> 代码

```java
//给定约瑟夫环的长度n，从第m个开始删，返回最后剩下的数的下标
    public int lastRemaining1(int n, int m) {
        if(n == 1) return 0;
        int res = 0; //初始化约瑟夫环长度为1时剩下的数就是0
        //迭代依次推出长度为2、3、4...n时该m下的约瑟夫环最后剩下的数
        for(int i = 2; i <= n; i++){
            // res = (res + m % i) % i;
            res = (res + m) % i;
        }
        return res;
    }
```

> 思考拓展

1. 这里的`n就相当于下标`了，以后给你一组数据并不是0 1 2 ....时我们也可以计算出最后留下来的元素`下标`从而确定最终留下来的元素。
2. 如果它并不是从第一个数开始算间隔求约瑟夫环的话（比如`从第X个数开始，以间隔m开始删除元素,F(n, x, m)`）我们可以先求出从第一个数开始以间隔m开始删除元素最后留下的元素下标F(n, 1, m),然后`F(n, x, m) = ( F(n, 1, m) + x ) % n`

# 67.  股票的最大利润-中等

> 题目

假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

> 思路1：动态规划

状态：dp[i]表示第i天卖出能获得的最大利润。

转移方程：dp[i] = dp[i - 1] + price[i] - price[i-1]。price[i]是第i天的股票价格

空间优化：由于只需要前一天卖出的最大利润，所以只需一个变量保存即可。

> 代码

```java
class Solution {
    public int maxProfit(int[] prices) {
        if(prices.length == 0) return 0;
        //如果知道了前一天卖出的最大利润和当天的股票价格，那么当天卖出的最大利润就确定了
        int pre = 0, now = 0, max = 0;
        for(int i = 1; i < prices.length; i++){
            now = prices[i] - prices[i - 1] + pre;
            if(now < 0) now = 0;
            pre = now;
            max = Math.max(now, max);
        }
        return max;
    }
}
```

# 68.  求1+2+…+n-中等

> 题目

求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

> 思路1：异常代替判断

虽然递归代替了循环，因为不能if，还要考虑递归到n == 0时怎么停止递归。

> 代码

```java
class Solution {
    public int sumNums(int n) {
        try{
            return n / n - 1 + n + sumNums(n - 1);
        }catch(Exception e){
            return 0;
        }
    }
}
```

> 思路2：短路来终止递归

使用“ && ”逻辑运算符的短路特性

> 代码

```java
class Solution {
    public int sumNums(int n) {
        int res = 0;
        boolean temp = n > 0 && (res = sumNums(n - 1) + n) > 0;
        return res;
    }
}
```

# 69.  不用加减乘除做加法-中等

> 题目

写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

> 思路:位运算

- ^ 亦或 ：相当于 `无进位的求和`。 想象10进制下的模拟情况：（如:19+1=20；无进位求和就是10，而非20；因为它不管进位情况）

- & 与 ：相当于`求每位的进位数`。先看定义：1&1=1；1&0=0；0&0=0；即都为1的时候才为1，正好可以模拟进位数的情况,还是想象10进制下模拟情况：（9+1=10，如果是用&的思路来处理，则9+1得到的进位数为1，而不是10，所以`要用<<1向左再移动一位`，这样就变为10了）；

这样公式就是：（a^b) ^ ((a&b)<<1) 即：每次无进位求 + 每次得到的进位数--------我们需要不断重复这个过程，直到进位数为0为止；

> 代码

```java
class Solution {
    public int add(int a, int b) {
        //都是1本位则为0并且进1即进位是1
        int benwei = a ^ b, jinwei = a & b;
        jinwei <<= 1;
        while(jinwei != 0){
            int temp = benwei;
            benwei = benwei ^ jinwei;
            jinwei = temp & jinwei;
            jinwei <<= 1;
        }
        return benwei;
    }
}
```

# 70.  构建乘积数组-简单

> 题目

- 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B[i] 的值是数组 A 中除了下标 i 以外的元素的积, 即 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。


> 思路:上三角、下三角

根据表格的主对角线（全为 11 ），可将表格分为 **上三角** 和 **下三角** 两部分。分别迭代计算下三角和上三角两部分的乘积，即可 **不使用除法** 就获得结果。

> 代码

```java
class Solution {
    public int[] constructArr(int[] a) {
        int len = a.length;
        if(len == 0) return new int[0];
        int[] arr1 = new int[len], arr2 = new int[len];
        //下三角
        arr1[0] = 1;
        for(int i = 1; i < len; i++){
            arr1[i] = arr1[i - 1] * a[i - 1];
        }
        //上三角
        arr2[len - 1] = 1;
        for(int i = len - 2; i >= 0; i--){
            arr2[i] = arr2[i + 1] * a[i + 1];
        }
        int[] res = new int[len];
        for(int i = 0; i < len; i++){
            res[i] = arr1[i] * arr2[i];
        }
        return res;
    }
}
```

# 71、把字符串转换成整数-中等

> 题目


写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似的库函数。

 

首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。

当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。

该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。

注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。

在任何情况下，若函数不能进行有效的转换时，请返回 0。

**说明：**

假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231, 231 − 1]。如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 

> 思路:注意条件即可

1. 去掉头尾空格
2. 第一个字符可以说正负号（只能有一个）
3. 转换到第一个非数字结束
4. 整数范围

> 代码

```java
class Solution {
    public int strToInt(String str) {
       str = str.trim();
       int flag = 1;
       long res = 0;
       for(int i = 0; i < str.length(); i++){
           char ch = str.charAt(i);
           if(ch == '-' && i == 0){
               flag = -1;
               continue;
           }
           if(ch == '+' && i == 0){
               continue;
           }
           if(ch >= '0' && ch <= '9'){
               res = res * 10 + ch - '0';
               if(res * flag > Integer.MAX_VALUE) return Integer.MAX_VALUE;
               if(res * flag < Integer.MIN_VALUE) return Integer.MIN_VALUE;
           }else{
               break;
           }
       }
       return (int)res * flag;
    }
}
```

# 72.  二叉搜索树的最近公共祖先-简单

> 题目

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

**说明:**

- 所有节点的值都是唯一的。
- p、q 为不同节点且均存在于给定的二叉树中。

> 思路1:常规递归

由于是二叉搜索树，所以如果两个结点都大于当前结点只需要在右子树去找，反之只需要去左子树去找。

只要两个结点p、q一个在当前结点左一个在右或一个是当前结点本身那么当前结点就是双方最近祖先

> 代码

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || root == p || root == q) return root;
       TreeNode left = lowestCommonAncestor(root.left, p, q);
       TreeNode right = lowestCommonAncestor(root.right, p, q);
       if(left == null && right == null) return null;
       if(left != null && right != null) return root;
       if(left == null) return right;
       else return left;
    }
}
```

> 思路:利用二叉树的性质

由于是二叉搜索树，所以如果两个结点都大于当前结点只需要在右子树去找，反之只需要去左子树去找。

只要两个结点p、q一个在 当前结点左  一个在右    或一个是当前结点本身那么当前结点就是双方最近祖先

> 代码

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        //题目说p、q一定存在，所以不用判断边界
        if(p.val > root.val && q.val > root.val) return lowestCommonAncestor(root.right, p, q);
        if(p.val < root.val && q.val < root.val) return lowestCommonAncestor(root.left, p, q);
        return root;
    }
}
```

# 73.  二叉树的最近公共祖先-简单

> 题目

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

**说明:**

- 所有节点的值都是唯一的。
- p、q 为不同节点且均存在于给定的二叉树中。

> 思路1:递归

由于不是是二叉搜索树，

> 代码

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || root == p || root == q) return root;
       TreeNode left = lowestCommonAncestor(root.left, p, q);
       TreeNode right = lowestCommonAncestor(root.right, p, q);
       if(left == null && right == null) return null;
       if(left != null && right != null) return root;
       if(left == null) return right;
       else return left;
    }
}
```

> 拓展思考：对于上面两道题：如果给出的p、q有可能不存在树中会是怎样的情况？

对于二叉搜索树来说肯定就不能用老套的简单判断大小来断定了。

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        //如果p、q的大小是在当前结点的两边但有可能p、q不存在，所以不行
        if(p.val > root.val && q.val > root.val) return lowestCommonAncestor(root.right, p, q);
        if(p.val < root.val && q.val < root.val) return lowestCommonAncestor(root.left, p, q);
        return root;
    }
}
```

对于普通二叉树分析

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        //这样如果给了两个结点一个存在一个不存在那么也会返回存在树中的这个结点，显然这不是两个结点的祖先
        if(root == null || root == p || root == q) return root;
       TreeNode left = lowestCommonAncestor(root.left, p, q);
       TreeNode right = lowestCommonAncestor(root.right, p, q);
       if(left == null && right == null) return null;
       if(left != null && right != null) return root;
       if(left == null) return right;
       else return left;
    }
}
```

