# 1.数组中重复的数字

> 题目

找出数组中重复的数字。


在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

> 思路：大小不超过数组下标

> 代码

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
        int len = nums.length;
        /**
        *长度为n的数组里存满了大小在0-n-1的数，如果不包含重复的数字那么每个数字i就刚好在
        *数组下标i处，如果下标i处的数据和它相等直接返回
        */
        for(int i = 0; i < len; i++){
            if(nums[i] != i){
                if(nums[i] == nums[nums[i]])
                return nums[i];
                int temp = nums[i];
                nums[i] = nums[temp];
                nums[temp] = temp;
                i--;//当前位置如果和其它位置交换了下次还需要判断当前位置
            }
        }
        return -1;
    }
}
```

# 2.二维数组中的查找

> 题目

在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

> 思路：从左下角或右上角的特殊性考虑

左下角或右上角的数都比左边大右边小

> 代码

```java
class Solution {
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if(matrix.length == 0) return false;
        //从右上角开始
        int j = matrix[0].length-1, i = 0;
        while(i < matrix.length && j >= 0){
            if(matrix[i][j] == target) return true;
            //大于就往左边走，小于就往下面走
            if(matrix[i][j] > target) j--;
            else i++;
        }
        return false;
    }
}
```

# 3.替换空格

> 题目

请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

> 思路：没感觉到这题想考些什么，new String(arr, 0, size)这个吗？

> 代码

```java
class Solution {
    public String replaceSpace(String s) {
        return s.replaceAll(" ", "%20");
    }
}
```

# 3. 从尾到头打印链表

> 题目

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

> 思路：栈或递归自己选

> 代码

```java
class Solution {
    List<Integer> list = new ArrayList<>();
    public int[] reversePrint(ListNode head) {
        reversePrint2(head);
        int arr[] = new int[list.size()];
        int i = 0;
        for(Integer num:list){
            arr[i++] = num;
        }
        return arr;
    }
    public void reversePrint2(ListNode head){
        if(head == null) return ;
        reversePrint(head.next);
        list.add(head.val);
    }
}
```

# 4. 重建二叉树

> 题目

输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

> 思路：与leetcode 105题一样

> 代码

```java
class Solution {
    HashMap<Integer, Integer> index = new HashMap<>();
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        for(int i = 0; i < inorder.length; i++){
            index.put(inorder[i], i);
        }
        return buildTree(preorder, inorder, 0, preorder.length-1, 0, inorder.length-1);
    }
     public TreeNode buildTree(int[] preorder, int[] inorder, int preorderBegin, int preorderEnd, int inorderBegin, int inorderEnd) {
         if(inorderEnd - inorderBegin < 0) return null;
         //前序的第一个值即为根结点
         TreeNode root = new TreeNode(preorder[preorderBegin]);
         if(inorderEnd == inorderBegin) return root;
         int rootIndex = index.get(preorder[preorderBegin]);
         int size = rootIndex - inorderBegin;//左子树结点个数
         root.left = buildTree(preorder, inorder, preorderBegin+1, preorderBegin+size, inorderBegin, rootIndex-1);
         root.right = buildTree(preorder, inorder, preorderBegin+size+1, preorderEnd, rootIndex+1, inorderEnd);
         return root;
    }
}
```

# 5. 用两个栈实现队列]

> 题目

用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )



> 思路：倒过去倒过来

> 代码

```java
Deque<Integer> stack1;
    Deque<Integer> stack2;
    public CQueue() {
        this.stack1 = new ArrayDeque<>();
        this.stack2 = new ArrayDeque<>();
    }
    
    public void appendTail(int value) {
        while(!stack2.isEmpty()){
            stack1.push(stack2.pop());
        }
        stack1.push(value);
    }
    
    public int deleteHead() {
        while(!stack1.isEmpty()){
            stack2.push(stack1.pop());
        }
        if(stack2.isEmpty()) return -1;
        else return stack2.pop();
    }
```



# 6. 斐波那契数列

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



# 7. 青蛙跳台阶问题

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

# 8. 旋转数组的最小数字

> 题目

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。

> 思路：二分

中间位置和左右边位置元素比较大小，由于时旋转数组，所以如果中间的元素大于最右边的，那么最小值一定在右边，如果小于则在左边。

注意：

- 与最右边等于时无法二分，只能right--
- 与最右边小于时由于中间位置元素可能刚好是最小的，所以次数right=mid，而不是=mid-1

> 代码

```java
class Solution {
    public int minArray(int[] numbers) {
        int left = 0, right = numbers.length-1;
        if(right == 0) return numbers[0];
        while(left < right){
            int mid = (left + right) / 2;
            if(numbers[mid] > numbers[right]){
                left = mid+1;
            }else{
                if(numbers[mid] < numbers[right]){
                    right = mid;
                }else{
                    right--;
                }
            }

        }
        return numbers[left];
    }
}
```

# 9. 矩阵中的路径

> 题目

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。

[["a","b","c","e"],
["s","f","c","s"],
["a","d","e","e"]]

但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。

> 思路：深度优先搜索+剪枝

一个位置可以走四个方向，但能不能走需要满足几个条件：

- 下一个节点不是边界
- 没有被遍历过
- 下个节点字符等于word下个字符

走进一个位置将该位置标记为遍历过，走出这个位置再取消标记。



> 代码

```java
class Solution {
    public boolean exist(char[][] board, String word) {
        if(board.length == 0 || word.length() == 0) return false;
        int height = board.length, len = board[0].length;
        boolean res = false;
        if(board.length * board[0].length < word.length()) return false;
        int trace[][] = new int[height][len];
        for(int i = 0; i < height; i++){
            for(int j = 0; j < len; j++){
                if(res) return res;
                //第一个字符相等才进入
                if(board[i][j] == word.charAt(0))
                res = exist(board, i, j, word, 1, trace);
            }
        }
        return res;
    }
    //i、j用来定位当前位置，index表示匹配到字符串的哪一个位置了，trace表示记录是否被遍历过
    public boolean exist(char[][] board, int i, int j, String word, int index, int[][] trace) {
        int height = board.length, len = board[0].length;
        if(index >= word.length()) return true;
        boolean res = false;
        //标记当前位置已经来过
        trace[i][j] = 1;
        //满足下一个节点不是边界、没有被遍历过、下个节点字符等于word下个字符三个条件才遍历下个节点
        if(i-1 >= 0 && trace[i-1][j] != 1 && board[i-1][j] == word.charAt(index)){
           res = exist(board, i-1, j, word, index+1, trace);
        }
        if(res) return res;
        if(i+1 < height && trace[i+1][j] != 1 && board[i+1][j] == word.charAt(index)){
            res = exist(board, i+1, j, word, index+1, trace);
        }
        if(res) return res;
        if(j-1 >= 0 && trace[i][j-1] != 1 && board[i][j-1] == word.charAt(index)){
             res = exist(board, i, j-1, word, index+1, trace);
        }
        if(res) return res;
        if(j+1 < len && trace[i][j+1] != 1 && board[i][j+1] == word.charAt(index)){
            res = exist(board, i, j+1, word, index+1, trace);
        }
        //剪枝回来时要把标记清楚
        trace[i][j] = 0;
        return res;
    }
}
```

# 10. 机器人的运动范围

> 题目

地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

 

> 思路：深度优先搜索

把每个方向能走的格子全部加起来，不像上一题是一条路径。但也要为走过的格子标记，只是不用取消标记

注意：

- 是下标各位上的数之和，不是下标简单加起来

- 是能够到达多少个格子，不是只走一次最多能到多少个格子。



> 代码

```java
class Solution {
    public int movingCount(int m, int n, int k) {
        if(k == 0) return 1;
        int[][] trace = new int[m][n];
        return movingCount(m, n, 0, 0, k, trace);
    }
    public int movingCount(int height, int len, int i, int j, int k, int[][] trace) {
        trace[i][j] = 1;
        int count = 0;
        if(i-1 >= 0 && trace[i-1][j] != 1 && getSum(i-1, j) <= k){
            count += movingCount(height, len, i-1, j, k, trace);
        }
        if(i+1 < height && trace[i+1][j] != 1 && getSum(i+1, j) <= k){
            count += movingCount(height, len, i+1, j, k, trace);
        }
        if(j-1 >= 0 && trace[i][j-1] != 1 && getSum(i, j-1) <= k){
            count += movingCount(height, len, i, j-1, k, trace);
        }
        if(j+1 < len && trace[i][j+1] != 1 && getSum(i, j+1) <= k){
            count += movingCount(height, len, i, j+1, k, trace);
        }
        // trace[i][j] = 0;因为是四个方向的加起来，不再取消标记
        //四个方向加起来再加上当前自己这个格子
        return count+1;
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

# 11. 剪绳子

> 题目

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

 

> 思路：每一段接近3乘积最大

> 代码

```java
class Solution {
    public int cuttingRope(int n) {
        if(n < 4) return n-1;
        int a = n / 3, b = n % 3;
        if(b == 0) return (int)Math.pow(3, a);
        if(b == 1) return (int)Math.pow(3, a-1) * 4;
        return (int)Math.pow(3, a) * 2;
    }
}
```

# 12. 剪绳子2

> 题目

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

 

> 思路：每一段接近3乘积最大

> 代码

```java
class Solution {
    public int cuttingRope(int n) {
        if(n < 4) return n - 1;
        int p = 1000000007;
        long res = 1L;
        while(n > 4){
            //每一次计算都要取余
            res = (res * 3) % p;
            n -= 3;
        }
        return (int)(n * res % p);
    }
}
```

# 12.二进制中1的个数

> 题目

请实现一个函数，输入一个整数（以二进制串形式），输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。

> 思路：常规计算二进制位，主要考虑负数

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

> 思路:目标值与1进行按位与运算

> 代码

```java
public class Solution {
    public int hammingWeight(int n) {
        int res = 0;
        while(n != 0) {
            res += n & 1;//得到最右边的二进制位
            n >>>= 1;
        }
        return res;
    }
}
```



# 12.数值的整数次方

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
        long b = n;//int最小负数转化为正数会超过范围
        if(b < 0){
            b *= -1;
            x = 1/ x;
        }
        while(b > 0){//最后一个数右移后b等于0
            long r = b & 1;//得出最右边二进制位数
            if(r == 1) res *= x;
            x = x * x;//x依次为：x的1次方、2次方、4、8、16次方.....
            b >>=1;//右移一位，遍历从左往右下一个二进制位
        }
        return res;
    }
}
```

# 13.打印从1到最大的n位数-简单

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
static char[]  loop = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
    static ArrayList<String> numbers = new ArrayList<>();
    public static void main(String args[]) {
        dfs("", 0, 2);
        for(String num:numbers){
            System.out.println(num);
        }
    }
    public static void dfs(String num, int index, int n){
        //回溯到第n位将该数存进列表中
        if(index == n){
            numbers.add(num);
            return;
        }
        for(char ch:loop){
            dfs(num+ch, index+1, n);
        }
    }
```

# 14.删除链表的节点-简单

> 题目

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点。

> 思路：递归、迭代随便

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

# 15.正则表达式匹配-困难

> 题目

请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。

> 思路：动态规划（自顶向下）

> 代码

```java
class Solution {
    public boolean isMatch(String s, String p) {
        return isMatch(s, p, s.length()-1, p.length()-1);
    }
    public boolean isMatch(String s, String p, int index1, int index2) {
        if(index2 < 0 && index1 >= 0) return false;//正则表达式字符串匹配完了，但目标字符串还没完，显然不匹配
        if(index1 < 0 && index2 < 0) return true;//两个字符串都匹配完了，说明能匹配
        //目标字符串匹配完了还可以和正则表达式的 字母+星号 进行匹配，相当于星号前的字母匹配0个
        if(index1 < 0){
            //正则表达式字符串是 字母+星号 就可以继续匹配，不是就显然不行
            if(p.charAt(index2) == '*')
            return isMatch(s, p, index1, index2 - 2);
            else
            return false;
        }
        char ch = p.charAt(index2);
        //如果不等于*号直接就看当前字符是否相互匹配和双方下一轮递归匹配
        if(ch != '*'){
            return (ch == '.' || ch == s.charAt(index1)) && isMatch(s, p, index1 - 1, index2 - 1);
        }else{
            //等于星号要判断星号后面的数是否匹配，不匹配直接当作星号前的字母匹配0个，再进行双方下一轮递归匹配
            //如果匹配则可以选择当前字符不进行匹配和进行匹配
            char ch2 = p.charAt(index2 - 1);
            if(ch2 == s.charAt(index1) || ch2 == '.'){
                return isMatch(s, p, index1 - 1, index2) || isMatch(s, p, index1, index2 - 2);
            }else{
                return isMatch(s, p, index1, index2 - 2);
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
        int m = s.length();
        int n = p.length();

        boolean[][] f = new boolean[m + 1][n + 1];
        f[0][0] = true;
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p.charAt(j - 1) == '*') {
                    //星号前的字母匹配0个
                    f[i][j] = f[i][j - 2];
                    //匹配一个或不匹配
                    if (matches(s, p, i, j - 1)) {
                        f[i][j] = f[i][j] || f[i - 1][j];
                    }
                } else {
                    //直接再进行双方下一轮递归匹配
                    if (matches(s, p, i, j)) {
                        f[i][j] = f[i - 1][j - 1];
                    }
                }
            }
        }
        return f[m][n];
    }

    public boolean matches(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        }
        if (p.charAt(j - 1) == '.') {
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }
}
```

# 16.表示数值的字符串-困难

> 题目

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"-1E-16"、"0123"都表示数值，但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是。

> 思路：我是傻X

‘.’出现正确情况：只出现一次，且在e的前面

‘e’出现正确情况：只出现一次，且出现前有数字

‘+’‘-’出现正确情况：只能在开头和e后一位

> 代码

```java
class Solution {
    public boolean isNumber(String s) {
        if (s == null || s.length() == 0) return false;
        //去掉首尾空格
        s = s.trim();
        boolean numFlag = false;
        boolean dotFlag = false;
        boolean eFlag = false;
        for (int i = 0; i < s.length(); i++) {
            //判定为数字，则标记numFlag
            if (s.charAt(i) >= '0' && s.charAt(i) <= '9') {
                numFlag = true;
                //判定为.  需要没出现过.并且没出现过e
            } else if (s.charAt(i) == '.' && !dotFlag && !eFlag) {
                dotFlag = true;
                //判定为e，需要没出现过e，并且出过数字了
            } else if ((s.charAt(i) == 'e' || s.charAt(i) == 'E') && !eFlag && numFlag) {
                eFlag = true;
                numFlag = false;//为了避免123e这种请求，出现e之后就标志为false
                //判定为+-符号，只能出现在第一位或者紧接e后面
            } else if ((s.charAt(i) == '+' || s.charAt(i) == '-') && (i == 0 || s.charAt(i - 1) == 'e' || s.charAt(i - 1) == 'E')) {

                //其他情况，都是非法的
            } else {
                return false;
            }
        }
        /**
        首先不管哪种情况都必须要有数字，没有数字肯定不对，其次对于存在 e 的情况，接收到 e 时已经判断前面有数字了，并将 			numFlag 重置，意味着当接收到 e 后，如果后面没有数字（没有将 numFlag 重新标记为真），那么也是不对的。
        */
        return numFlag;
    }
}
```

# 17.调整数组顺序使奇数位于偶数前面-简单

> 题目

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

> 思路：快排思想，头尾双指针

> 代码

```java
class Solution {
    public int[] exchange(int[] nums) {
        int len = nums.length;
        if(len < 2) return nums;
        int left = 0, right = len - 1;
        while(left < right){
            while(left < right){
                if(nums[left] % 2 == 0){
                    int temp = nums[left];
                    nums[left] = nums[right];
                    nums[right] = temp;
                    break;
                }
                left++;
            }
            while(right > left){
                if(nums[right] % 2 != 0){
                    int temp = nums[left];
                    nums[left] = nums[right];
                    nums[right] = temp;
                    break;
                }
                right--;
            }
            
        }
        return nums;
    }
}
```

# 18.链表中倒数第k个节点-简单

> 题目

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。

> 思路：双指针，遍历一遍即可

> 代码

```java
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        if(head == null) return null;
        ListNode left = head, right = head;
        int cnt = 1;
        while(cnt < k){
            right = right.next;
            cnt++;
        }
        while(right.next != null){
            right = right.next;
            left = left.next;
        }
        return left;
    }
}
```



# 19.反转链表-简单

> 题目

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

> 思路：递归或前后指针迭代

> 代码

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null || head.next == null)
            return head;
        ListNode Newhead = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return Newhead;
    }
}
```



# 20.合并两个排序的链表-简单

> 题目

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

> 思路：归并思想或递归

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



# 21.树的子结构-中等

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
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if(A == null || B== null) return false;
        //找到与子结构树根节点相等的节点就判断能不能匹配，不能匹配就继续向左右子树找
        if(A.val == B.val && isSub(A, B)) return true;
        else return isSubStructure(A.left, B) || isSubStructure(A.right, B);
    }
    //和判断两棵树是否完全相等稍微有点差别
    public boolean isSub(TreeNode A, TreeNode B){
        if(B == null) return true;//B为null了不管A是否为null显然匹配成功
        if(A == null) return false;//A为null但B不为null显然匹配失败

        if(A.val == B.val){
            return isSub(A.left, B.left) && isSub(A.right, B.right);
        }else{
            return false;
        }
    }
}
```



# 22. 二叉树的镜像-简单

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
    public TreeNode mirrorTree(TreeNode root) {
        if(root == null) return null;
        TreeNode left = root.left;//左边保存起，不然就被覆盖了
        root.left = mirrorTree(root.right);
        root.right = mirrorTree(left);
        return root;
    }
}
```

# 23 . 对称的二叉树-简单

> 题目

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

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
        if(root1 == null || root2 == null || root1.val != root2.val) return false;
        return isSymmetric(root1.left, root2.right) && isSymmetric(root1.right, root2.left);
    }
}
```

# 24.顺时针打印矩阵-简单

> 题目

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

> 思路：主要是考虑边界值

> 代码

```java
class Solution {
    public int[] spiralOrder(int[][] matrix) {
         if(matrix.length == 0) return new int[0];
        int height = matrix.length, len = matrix[0].length;     
        int[] res = new int[height * len];
        int resIndex = 0;
        int L = 0, H = 0, i = 0, j = 0; 
        while(resIndex < height * len){
            //左往右
            for(j = L; j < len-L && resIndex < height * len; j++){
                res[resIndex++] = matrix[i][j];
            }
            j--;//因为循环完j刚好超过边界，需要回到边界位置，后面的i--、j++、i++一样的道理
            //下往上
            for(i = i+1; i < height-H && resIndex < height * len; i++){
                res[resIndex++] = matrix[i][j];
            }
            i--;
            //这里是大于等于L，但下面是大于H不能等于H，因为从右往左时左边界的值还没有遍历，但从下往上时上边界的值已经被遍历过了
            //右往左
            for(j = j-1; j >= L && resIndex < height * len; j--){
                res[resIndex++] = matrix[i][j];
            }
            j++;
            //下往上
            for(i = i-1; i > H && resIndex < height * len; i--){
                res[resIndex++] = matrix[i][j];
            }
            i++;
            //H、L各自加一，表示已经饶了一圈了，下次起始边界应该加一（j=L），结束边界应该减一（j < len-L和i < height-H）
            H++;
            L++;
        }
        return res;
    }
}
```

# 25、包含min函数的栈-简单

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
        if(head == null) head = new Node(x, x);
        else head = new Node(x, Math.min(head.min, x), head);
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

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.min();
 */
```



# 25.栈的压入、弹出序列-中等

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
        int pushedLength = pushed.length, poppedLength = popped.length;
        if(pushedLength != poppedLength) return false;
        if(pushedLength < 2) return true;
        //将压栈元素的下标放在map里方便后面快速找到pop的值在pushed中的位置
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < pushedLength; i++){
            map.put(pushed[i], i);
        }
        //第一个肯定是下标减一，后面就不是了
        int index = map.get(popped[0])-1;
        for(int i = 1; i < poppedLength; i++){
            //如果不在改范围直接返回false
            if(!inArray(pushed, index, pushedLength-1, popped[i]))
                return false;
            //确定 ” 范围的开始下标 “，也就是前面第一个还没有弹出的元素下标
            index =  map.get(popped[i]) - 1;
            for(int j = index; j >= 0; j--){
                index = j;
                //不再已经弹出的元素中，说明找到了第一个还没有弹出的元素，直接break
                if(!inArray(popped, 0, i, pushed[j])) break;
            }
            // System.out.println(index);
        }
        return true;
    }
    //判断arr【beginIndex，endIndex】中是否含target
    public boolean inArray(int[] arr, int beginIndex, int endIndex, int target){
        if(beginIndex < 0) beginIndex = 0;
        for(int i = beginIndex; i <= endIndex; i++){
            if(arr[i] == target) return true;
        }
        return false;
    }
}
```

> 思路2：辅助栈

遍历push列表，一次压入栈中，在压入过程中判断栈顶元素是否和pop列表元素相等，相等就出栈至不相等，如果pop序列没错，则最后栈中元素肯定出栈完了。

> 代码

```java
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
       int pushedLength = pushed.length, poppedLength = popped.length;
       if(pushedLength < 2) return true;
       Deque<Integer> stack = new ArrayDeque<>();
       int j = 0;
       for(int num: pushed){
           stack.push(num);
           while(!stack.isEmpty() && stack.peek() == popped[j]){
               stack.pop();
               j++;
           }
       }

       return stack.isEmpty();
    }
   
}
```

# 26. 从上到下打印二叉树-中等

> 题目

从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

> 思路：层序遍历

> 代码

```java
class Solution {
    public int[] levelOrder(TreeNode root) {
        if(root == null) return new int[0];
        Deque<TreeNode> queue = new ArrayDeque<>();
        List<Integer> numbers = new ArrayList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode temp = queue.poll();
            numbers.add(temp.val);
            if(temp.left != null) queue.offer(temp.left);
            if(temp.right != null) queue.offer(temp.right);
        }
        int res[] = new int[numbers.size()];
        int i = 0;
        for(Integer num: numbers){
            res[i++] = num;
        }
        return res;
    }
}
```

# 27. 从上到下打印二叉树2-中等

> 题目

从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。每一层区分开来

> 思路：层序遍历

每一次循环得到当前队列的size，这size个元素就是一层的元素

> 代码

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Deque<TreeNode> queue = new ArrayDeque<>();
        List<List<Integer>> lists = new ArrayList<>();
        if(root == null) return lists;
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
            lists.add(list);
        }
        return lists;
    }
}
```

# 28. 从上到下打印二叉树3-中等

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

# 28. 二叉搜索树的后序遍历序列-中等

> 题目

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

> 思路1：递归

后续遍历最后一个元素为根节点，从前面开始遍历找到第一个大于根节点的值的位置，那么这个位置后面就是该根节点的右子树

所以在这位置后面的每一个元素都必须大于根节点。再递归验证左右子树，即该位置前面序列和后面序列

> 代码

```java
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        return verifyPostorder(postorder, 0, postorder.length - 1);
    }
    public boolean verifyPostorder(int[] postorder, int begin, int end) {
        //如果小于两个元素了肯定就是二次搜索树
        if(begin >= end) return true;
        //找到第一个大于根节点的值的位置，那么它后面就是该根节点的右子树，
        int mid = begin, root = postorder[end];
        while(postorder[mid] < root){
            mid++;
        }
        //如果后面有小于根节点的值就肯定不是二叉搜索树
        int i = mid + 1;
        while(i < end){
            if(postorder[i++] < root) return false;
        }
        return verifyPostorder(postorder, begin, mid - 1) && verifyPostorder(postorder, mid, end - 1);
    }
}
```

> 思路2：单调栈

将后序遍历反过来就是先遍历左节点的前序遍历。二叉搜索树的先遍历左节点的前序遍历一开始的节点值都是不停增大，一旦变小就说明进入某个节点的左节点了，后面遍历的节点值都必须小于这个“ 某个节点 ” 的值。这样就可以利用单调栈找到这 “ 某个节点 ”，然后比较后面的值，如果大于它就返回false。



与前面的思路比较：

- 前面是找到第一个大于根节点的值，该值后面的节点值都必须大于它，但这样一次只知道了当前根节点满足，左右子树的一些根节点不知道满不满足，所以然后再利用递归对后面和前面的树继续这样判断。

- 而单调栈的思路找到进入左边遍历前最小的元素（左边的根节点），后面节点的值都必须小于这个值，之所以不用向前面一样递归是因为在后续入栈过程中会同时更新进入下一个左边遍历前最小的元素（（左边的根节点）），就是相当于一直保证左边的节点值比左边根节点都要小

> 代码

```java
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        Deque<Integer> stack = new ArrayDeque<>();
        int prev = Integer.MAX_VALUE;
        for(int i = postorder.length - 1; i >= 0; i--){
            //后面的节点值必须小于“左节点的父节点”
            if(postorder[i] > prev) return false;
            while(!stack.isEmpty() && stack.peek() > postorder[i]){
                //左节点的父节点，
                prev = stack.pop();
            }
            stack.push(postorder[i]);
        }
        return true;
    }
}
```

# 29、二叉树中和为某一值的路径 -中等

> 题目

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

> 思路：递归

递归压栈时list在不停加元素，出栈时需要去除list对应元素

注意：必须是根结点到 ” **叶子结点** “ 。也就是只有到叶子节点了才判断sum是否等于目标值，其它地方的结点和等于目标值不算

> 代码

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        if(root == null) return res;
        List list = new ArrayList<>();
        pathSum(root, targetSum, 0, list);
        return res;
    }
    public void pathSum(TreeNode root, int targetSum, int sum, List list) {
        list.add(root.val);
        if(root.left == null && root.right == null){
            if(sum + root.val == targetSum){
                res.add(new ArrayList<>(list));//不能直接添加list到res中，需要new个新的list，将值复制进去
                return ;
            }
        }else{
            sum += root.val;
        }
        if(root.left != null){
            pathSum(root.left, targetSum, sum, list);
            list.remove(list.size()-1);//如果左边为null就不需要在list中移除元素
        }
        if(root.right != null){
            pathSum(root.right, targetSum, sum, list);
            list.remove(list.size()-1);//如果右边为null就不需要在list中移除元素
        }
        return ;
    }
}
```



# 29. 复杂链表的复制-中等

> 题目

请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

> 思路1：HashMap存放每一个老结点：新结点 键值对

先创建新结点只初始化值，并将老结点：新结点键值对加入map，然后再遍历老结点，新结点的next就是老结点的next对应的新结点，新结点的random就是老结点的random对应的新结点

> 代码

```java
class Solution { //HashMap实现
    public Node copyRandomList(Node head) {
        HashMap<Node,Node> map = new HashMap<>(); //创建HashMap集合
        Node cur=head;
        //复制结点值
        while(cur!=null){
            //存储put:<key,value1>
            map.put(cur,new Node(cur.val)); //顺序遍历，存储老结点和新结点(先存储新创建的结点值)
            cur=cur.next;
        }
        //复制结点指向
        cur = head;
        while(cur!=null){
            //得到get:<key>.value2,3
            map.get(cur).next = map.get(cur.next); //新结点next指向同旧结点的next指向
            map.get(cur).random = map.get(cur.random); //新结点random指向同旧结点的random指向
            cur = cur.next;
        }

        //返回复制的链表
        return map.get(head);


    }
}
```

> 思路2：空间复杂度O(1)的原地复制

看注解

> 代码

```java
class Solution {
    /**
    HashSet<Node> set = new HashSet<>();
    public Node copyRandomList(Node head) {
        if(head == null || set.contains(head)) return head;
        Node node = new Node(head.val);
        set.add(node);
        node.next = copyRandomList(head.next); 
        node.random = copyRandomList(head.random);
        return node;
    }
    */
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
}
```



# 30. 二叉搜索树与双向链表-中等

> 题目

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

> 思路1：前后指针

当前指针的左指针指向前面的结点，前面的结点的右指针指向当前结点。递归、栈都可以，不过这样做完后最左边的结点的左指针和最右边的结点的右指针还是指向null的，需要单独处理一下。

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

# 31. . 序列化二叉树-中等

> 题目

请实现两个函数，分别用来序列化和反序列化二叉树。

> 思路1：使用LinkedList实现的Queue接口的队列对象层序遍历

ArrayDeque对象当做队列使用将无法存储null值，LinkedList对象当做队列使用可以存放null值.

再根据层序遍历将其还原为二叉树。

> 代码

```java
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
        return res.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
       if("".equals(data) || data == null) return null;
       String[] list = data.substring(1, data.length()-1).split(",");
       //根结点为null
       if("null".equals(list[0])) return null;
       Queue<TreeNode> queue = new LinkedList<>();
       //先根据层序遍历顺序的第一个结点值创建第一个结点，放进队列中
       TreeNode res = new TreeNode(Integer.parseInt(list[0]));
       queue.offer(res);
       int i = 1;
       while(!queue.isEmpty()){
           //队头结点出队，根据列表顺序后两个建立出队结点的左右子节点
           TreeNode root = queue.poll();
           if(!"null".equals(list[i])){
               TreeNode left = new TreeNode(Integer.parseInt(list[i]));
               root.left = left;
               //同时将新建的左结点入队列
               queue.offer(left);
           }
           i++;
           if(!"null".equals(list[i])){
               TreeNode right = new TreeNode(Integer.parseInt(list[i]));
               root.right = right;
               //同时将新建的左结点入队列
               queue.offer(right);
           }
           i++;
       }
       return res;
    }
    
}

```

> 思路2：本题只要求能讲树序列化后能反序列化为原样就行，可使用DFS

序列化为前序遍历的字符串，然后将字符串依次放入队列，对队列进行递归建树。

> 代码

```java
public class Codec {

    // 将树序列化为前序遍历的字符串
    public String serialize(TreeNode root) {
        if(root == null) return "null";
        //序列化为前序遍历的字符串,以以逗号隔开
        return root.val + "," + serialize(root.left) + "," + serialize(root.right);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        //将前序序列字符串转化为对应的队列
        Queue<String> queue = new LinkedList<>(Arrays.asList(data.split(",")));
        return dfs(queue);
    }
    //前序序列在队列中按原顺序排队，依次出队递归建立二叉树
    public TreeNode dfs(Queue<String> queue){
        //队列最后一定是null字符串，即将为空时直接返回null了，所以不用管队列为空
        String val = queue.poll();
        if("null".equals(val)) return null;
        TreeNode root = new TreeNode(Integer.parseInt(val));
        root.left = dfs(queue);
        root.right = dfs(queue);
        return root;
    }
}
```

# 32. 字符串的排列-中等

> 题目

输入一个字符串，打印出该字符串中字符的所有排列。

 

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

> 思路：回溯

主要是需要考虑前面某个位置上的字符已经被选择过，那么那个位置上字符不能再被选择，这单可以利用一个set集合存放回溯过的字符下标，下次遍历回溯只能选择不再集合中位置上的字符，不过每次回溯完需要将前面加入set的下标删除，即“ 剪枝 ”。

不过还有一种情况，就是字符串里有重复字符，这时候会产生一样的排列，比如abcc，这是会有两个一样的下标排列0123和0132，他们都是abcc，所以需要再加一个map判断结果字符串是否重复

> 代码

```java
剪枝class Solution {

    ArrayList<String> list = new ArrayList<>();
    HashSet<Integer> set = new HashSet<>();//存放回溯过的字符下标
    HashSet<String> listSet = new HashSet<>();//判断结果字符串是否重复
    String str;
    public String[] permutation(String s) {
        if(s == null || s.equals("")) return new String[0];
        str = s;
        flashBack("", 0, s.length());
        return list.toArray(new String[0]);
    }
    void flashBack(String res, int cnt, int n){
        if(cnt == n){
            if(!listSet.contains(res)){
                list.add(res);
                listSet.add(res);
            }
            return ;
        }

        for(int i = 0; i < n; i++){
            if(!set.contains(i)){
                set.add(i);//只能选择不再集合中位置上的字符
                flashBack(res+str.charAt(i), cnt + 1, n);
                set.remove(i);//剪枝
            }
        }
    }
}
```



# 35. 二数组中出现次数超过一半的数字-简单

> 题目

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

 

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

> 思路：摩尔投票

数相同总数加1，不同则抵消，到最后的数一定是出现大于一半的数

> 代码

```java
class Solution {
    public int majorityElement(int[] nums) {
        //初始化出现剩余个数为0,
        int count = 0, res = 0;
        for(int num: nums){
            //如果剩余次数为0，res等于当前值
            if(count == 0) res = num;
            //如果和当前值相等当前res的剩余个数就加1，否则减一，减到0后就更新res的值
            count += (res == num ? 1:-1); 
        }
        //由于这个数肯定出现超过一半，所以最终遍历后剩余次数大于0的一定是这个数
        return res;
    }
}
```

# 36. 二最小的K个数-简单

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

# 37. 数据流中的中位数-简单

> 题目

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

> 思路：一个大根堆一个小根堆分别保存前一半小的元素和前一半大的元素。

> 代码

```java
class MedianFinder {

    //大根堆
    PriorityQueue<Integer> pq1 = new PriorityQueue<>(new Comparator<Integer>(){
        public int compare(Integer a, Integer b){
            return b - a;
        }
    });
    //小根堆
    PriorityQueue<Integer> pq2 = new PriorityQueue<>();

    /** initialize your data structure here. */
    public MedianFinder() {
        
    }
    
    public void addNum(int num) {
        //先放到大根堆pq1里面，之后依次循环着放
        if(pq1.size() == pq2.size()){
            //放入之前必须进过另一个堆，这样才能保证放入本堆中的元素是前一半小的，下面同理
            pq2.offer(num);
            pq1.offer(pq2.poll());
        }else{
            pq1.offer(num);
            pq2.offer(pq1.poll());
        }
    }
    
    public double findMedian() {
        int sum = pq1.size() + pq2.size();
        if(sum % 2 == 0){
           return (pq1.peek() + pq2.peek()) / 2.0;
        }else{
            return (double)pq1.peek();
        }
    }
}
```

# 38. 连续子数组的最大和-简单

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
        //prev表示以前面一个位置元素结尾的子数组的最大和
        //now表示现在位置元素结尾的子数组的最大和
        //每计算出一个位置结尾的子数组的最大和都更新最大值res,遍历完每个位置后结果就出来了
        int res = nums[0], prev = 0, now = 0;
        for(int num: nums){
            if(prev > 0){
                now = prev + num;
            }else{
                now = num;
            }
            prev = now;
            res = Math.max(res, now);
        }
        return res;
    }
}
```

# 39.  1～n 整数中 1 出现的次数-困难

> 题目

输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。

例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

> 思路：递归

[大佬题解](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/solution/javadi-gui-by-xujunyi/)

> 代码

```java
class Solution {
    public int countDigitOne(int n) {
        return f(n);
    }
    private int f(int n ) {
        if (n <= 0)
            return 0;
        String s = String.valueOf(n);
        int high = s.charAt(0) - '0';
        int pow = (int) Math.pow(10, s.length()-1);
        int last = n - high*pow;
        if (high == 1) {
            return f(pow-1) + last + 1 + f(last);
        } else {
            return pow + high*f(pow-1) + f(last);
        }
    }
}
```

# 41. 把数组排成最小的数-中等

> 题目

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

> 思路：排序

大小比较规则：num1+num2 < num2 + num1

> 代码

```java
class Solution {
    public String minNumber(int[] nums) {
        if(nums.length == 0) return "";
        Integer[] number = new Integer[nums.length];
        for(int i = 0; i < nums.length; i++){
            number[i] = nums[i];
        }
        Arrays.sort(number, new Comparator<Integer>(){
            public int compare(Integer a, Integer b){
                String str1 = a.toString();
                String str2 = b.toString();
                return (str1+str2).compareTo(str2+str1);
            }
        });
        String res = "";
        for(int num: number){
            res += num;
        }
        return res;
    }
}
```

# 42.  把数字翻译成字符串-中等

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
        if(num < 10) return 1;
        String number = num + "";
        int len = number.length();
        int dp[] = new int[len + 1];
        dp[0] = 1;
        dp[1] = 1;
        char pre = number.charAt(0);
        for(int i = 1; i < len; i++){
            int temp = Integer.parseInt(pre + "" + number.charAt(i));
            if(temp >= 0 && temp <= 25 && pre != '0') dp[i+1] = dp[i] + dp[i-1];
            else dp[i+1] = dp[i];
            pre = number.charAt(i);
        }
        return dp[len];
    }
}
```

# 42.  礼物的最大价值-中等

> 题目

在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

> 思路1：动态规划(自顶向下)

往右边走、与往下边走两个选择，选最大的即可。考虑到一些数据重复递归计算耗费大量时间，可以定义个状态数组存储已经计算过的到达该位置能得到的最大权重

> 代码

```java
class Solution {
    int[][] dp;//保存计算过的到达该位置能得到的最大权值
    public int maxValue(int[][] grid) {
        if(grid.length == 0) return 0;
        dp = new int[grid.length+1][grid[0].length+1];
        return maxValue(grid, 0, 0);
    }
    public int maxValue(int[][] grid, int i, int j) {
        if(i >= grid.length || j >= grid[0].length) return 0;
        int res1 = 0, res2 = 0;
        //如果下个位置已经计算过就不再递归重复计算了
        if(dp[i+1][j] == 0){
            res1 = maxValue(grid, i+1, j);
            dp[i+1][j] = res1;
        }else{
            res1 = dp[i+1][j];
        }
        if(dp[i][j+1] == 0){
            res2 = maxValue(grid, i, j+1);
            dp[i][j+1] = res2;
        }else{
            res2 = dp[i][j+1];
        }
        return grid[i][j] + Math.max(res1, res2);
    }
}
```

> 思路2：动态规划(自底向上)

状态dp[i]\[j]：到达下标i、j这个位置能得到的最大权值。

状态转移：dp[i]\[j] = dp[i-1]\[j]与dp[i]\[j-1]的最大值加上dp[i]\[j]

> 代码

```java
class Solution {
    public int maxValue(int[][] grid) {
        if(grid.length == 0) return 0;
        int row = grid.length, col = grid[0].length;
        for(int i = 0; i < row; i++){
            for(int j = 0; j <col; j++){
                int temp1 = 0, temp2 = 0;
               	//考虑边界
                if(i - 1 >= 0) temp1 = grid[i-1][j];
                if(j - 1 >= 0) temp2 = grid[i][j-1];
                grid[i][j] = Math.max(temp1, temp2) + grid[i][j];
            }
        }
        return grid[row-1][col-1];
    }
}
```

# 43.最长不含重复字符的子字符串-中等

> 题目

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度？

> 思路1：滑动窗口

前后指针left与right，没有遇到重复的话left保持不动right一直向前走，他们两个之间的距离就是当前无重复子字符串的长度；一旦遇到重复的，left指针就向右边移动到重复字符位置的下一个字符位置，然后继续保持不动right继续向前走直到遇到下一个重复的。不停循环到right到字符串长度，循环过程中更新maxLen即可。

那怎样确定是否出现重复字符并且确定重复字符的位置呢？可以使用各种各样的方式实现：

1. 队列：遇到重复的就将元素出队到不再重复，此时队头就刚好是重复字符的下一个字符。
2. HashMap：key为字符，value为字符位置
3. 数组实现HashMap：由于是ASCLL字符，所有可以定义个数组实现HashMap，下标为字符，值为字符位置。

> 代码1:使用队列

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        //用链表实现队列，队列是先进先出的
        Queue<Character> queue = new LinkedList<>();
        int res = 0;
        for (char c : s.toCharArray()) {
            while (queue.contains(c)) {
                //如果有重复的，队头出队
                queue.poll();
            }
            //添加到队尾
            queue.add(c);
            res = Math.max(res, queue.size());
        }
        return res;
    }
}
```

> 代码2：使用数组

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int maxLen = 0;
        int index[] = new int[300];//自动初始化为0
        for(int i = 0, j = 0; i < s.length(); i++)
        {
            //如果有重复元素j就等于重复元素的位置，否则就不变（index数组没有记录默认为0）
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

# 44.丑数-中等

> 题目

我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

> 思路：没得

8=2*2\*2;质因子为2；12=2\*2\*3质因子为2和3。任意一个丑数都是由小于它的某一个丑数2\*3或者\*5得到的，我们需要按从小到大的顺序得到n个丑数，从第一个丑数1开始可以知道1\*2、1\*3、1\*5三个丑数，选最小的即为第二个丑数，第三个丑数又是从第二个丑数\*2、1\*3、1\*5这三个里面选最小的。一直循环下去.......

> 代码:

```java
class Solution {
    public int nthUglyNumber(int n) {
        int p2 = 0, p3 = 0, p5 = 0;
        int ugly[] = new int[n];
        ugly[0] = 1;
        for(int i = 1; i < n; i++){
            //选最小的
            ugly[i] = Math.min(Math.min(ugly[p2] * 2, ugly[p3] * 3), ugly[p5] * 5);
            //都要判断一次因为可能会产生重复的值，如果重复就要跳过
            if(ugly[i] == ugly[p2] * 2) p2++;
            if(ugly[i] == ugly[p3] * 3) p3++;
            if(ugly[i] == ugly[p5] * 5) p5++;
        }
        return ugly[n-1];
    }
}
```

# 46.第一个只出现依次的字符-简单

> 题目

在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

> 思路：LinkedHashMap

key为字符，value为该字符是否重复

> 代码:

```java
class Solution {
    public char firstUniqChar(String s) {
       Map<Character, Boolean> dic = new HashMap<>();

        char[] array = s.toCharArray();

        for (char c : array) {
            //如果重复了就是false
            dic.put(c, !dic.containsKey(c));
        }

        for (char c : array) {
            if (dic.get(c)) return c;
        }

        return ' ';
    }
}
```

# 47.数组中的逆序对-困难

> 题目

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。。

> 思路：归并排序

数组左边区域与右边区域合并的过程中如果左边某个数大于右边的某个数，那么此时左边的这个数和该数后面的所有数(在左边区域内)都可以与右边这个数构成逆序对，所有此时执行sum += mid - i + 1;

> 代码:

```java
class Solution {
    int sum = 0;
    public int reversePairs(int[] nums) {
        int len = nums.length;
        if(len < 2) return 0;
        mergeSort(nums, 0, len - 1, new int[len]);
        return sum;
    }
    public void mergeSort(int arr[], int left, int right, int temp[]){
        if(left == right) return;
        int mid = left + (right- left) / 2;
        mergeSort(arr, left, mid, temp);
        mergeSort(arr, mid+1, right, temp);
        mergeSort2(arr, left, right, temp);
    }
    public void mergeSort1(int[] arr, int left, int right, int[] temp){
        if(left == right) return;
        int mid = left + (right - left) / 2;
        for(int i = left; i <= right; i++){
            temp[i] = arr[i];
        }
        for (int i = left, j = mid + 1, k = left; k <= right; k++) {
            if(i == mid + 1){
                arr[k] = temp[j++];
            }else if(j == right + 1){
                arr[k] = temp[i++];
            }else if(temp[i] <= temp[j]){
                arr[k] = temp[i++];
            }else{
                arr[k] = temp[j++];
                sum += mid - i + 1; //前面的都大于它，都是一个逆序对
            }
        }
    }
    public void mergeSort2(int[] arr, int left, int right, int[] temp){
        if(left == right) return;
        int mid = left + (right - left) / 2;
        int i = left, j = mid + 1, k = left;

        while(i <= mid && j <= right){
            if(arr[i] <= arr[j]) temp[k++] = arr[i++];
            else {
                temp[k++] = arr[j++];
                sum += mid - i + 1;
            }
        }

        while(i <= mid) temp[k++] = arr[i++];
        while(j <= right) temp[k++] = arr[j++];

        for(int x = left; x <= right; x++){
            arr[x] = temp[x];
        }
    }
}
```



# 47.在排序数组中查找数字 I-简单

> 题目

统计一个数字在排序数组中出现的次数。

> 思路：二分

找到目标值左右两边的边界下标

> 代码:

```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length-1, mid = 0;
        while(left <= right){
            mid = left + (right - left) / 2;
            if(nums[mid] >= target) right = mid - 1;
            else left = mid + 1;
        }
        if(right + 1 < nums.length && nums[right+1] != target) return 0;
        //左边界下标，该处的值小于目标值或下标为-1
        int before = right;
        //left可以不用变了
        right = nums.length - 1;
        while(left <= right){
            mid = left + (right - left) / 2;
            if(nums[mid] > target) right = mid - 1;
            else left = mid + 1;
        }
        //0 2 2 3 => 3 - 0 - 1
        return left - before - 1;
    }
}
```

# 48. 0～n-1中缺失的数字-简单

> 题目

一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

> 思路：二分

长度为n的数组升序存储[0, n]范围的数字，相当于是一个值与下标完全一样的长为n+1的数组从中拿走了一个值，拿走的这个值的位置前面的所有数组值与下标仍然一样，只是后面的数组值与下标不一样（大1），所以我们可以通过二分去找到第一个值与下标不一样的元素位置或最后一个值与下标一样的元素位置。

> 代码:

```java
class Solution {
    public int missingNumber(int[] nums) {
        int len = nums.length;
        if(nums[len - 1] == len - 1) return len;
        int left = 0, right = len - 1, mid = 0;
        while(left <= right){
            mid = left + (right - left) / 2;
            if(nums[mid] != mid) right = mid - 1;
            else left = mid + 1;
        }
        //最后right即为最后一个值与下标一样的元素位置。它的下一个就是被拿走的元素
        return right + 1;
    }
}
```

# 49.二叉搜索树的第k大节点-简单

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
    int res = -1, i = 1;
    public int kthLargest(TreeNode root, int k) {
        f(root, k);
        return res;
    }
     public void f(TreeNode root, int k) {
        if(root == null) return;
        f(root.right, k);
        if(i == k) res = root.val;
        i++;
        f(root.left, k);
    }
     public int kthLargest2(TreeNode root, int k) {
        int i = 1;
        ArrayDeque<TreeNode> stack = new ArrayDeque<>();
        while(!stack.isEmpty() || root != null){
            while(root != null){
                stack.push(root);
                root = root.right;
            }
            TreeNode top = stack.pop();
            if(i == k) return top.val;
            i++;
            root = top.left;
        }
        return -1;
    }
}
```

# 50.二叉树的深度-简单

> 题目

输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

> 思路：递归

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

# 51.平衡二叉树-简单

> 题目

输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

> 思路：	和计算深度一样，不过为了提高效率需要提前阻断

提前阻断：一个子节点不是平衡的那么他上面的所有节点就不用计算深度了

> 代码:

```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        return recur(root) != -1;
    }

    private int recur(TreeNode root) {
        if (root == null) return 0;
        int left = recur(root.left);
        if(left == -1) return -1;
        int right = recur(root.right);
        if(right == -1) return -1;
        return Math.abs(left - right) < 2 ? Math.max(left, right) + 1 : -1;
    }
}
```

# 52.数组中数字出现的次数-中等

> 题目

一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

> 思路：看代码



> 代码

```java
class Solution {
    public int[] singleNumbers(int[] nums) {
        /**
        /*1、得到没有重复的两个数的异或结果
        /*2、找出这两个数在二进制位中不同某一位，也就是res1二进制位上为1的位置
        /*3、找到不同的这个位置后我们在遍历数组异或运算就可以把另一个不重复的数区分出来不参加异或运算
        /*   从而最后的运算结果就是自己了，而不是两个数的异或结果了
        */

        //1、得到没有重复的两个数的异或结果
        int res1 = 0;
        for(int num: nums){
            res1 ^= num;
        }
        //2、找出这两个数在二进制位中不同某一位，也就是res1任意一个二进制位上为1的位置
        //我们这里是从二进制位右往左找第一个为1的位置（你找其它地方为1的位置也行）
        int cnt = 0;
        while((res1 & 1) == 0){
            cnt++;
            res1 >>= 1;
        }
        //前面的cnt就是从右往左第cnt+1个位置上为1，这里flag就是该位上为1，其它位全为0的数，目的是为了后面遍历数组时容易区分两个不重复的数
        int flag = 1 << cnt;
        
        //3、找到不同的这个位置后我们在遍历数组异或运算就可以把另一个不重复的数区分出来不参加异或运算
        //   从而最后的运算结果就是自己了，而不是两个数的异或结果了
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

# 53.数组中数字出现的次数2-中等

> 题目

在一个数组 `nums` 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

> 思路：状态机、位运算：[大佬思路](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/solution/mian-shi-ti-56-ii-shu-zu-zhong-shu-zi-chu-xian-d-4/)

> 代码

```java
class Solution {
    public int singleNumber(int[] nums) {
        int ones = 0, twos = 0;
        for(int num : nums){
            ones = ones ^ num & ~twos;
            twos = twos ^ num & ~ones;
        }
        return ones;
    }
}
```

# 54.和为s的两个数字-简单

> 题目

输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

> 思路：排好了序可以利用头尾双指针

> 代码

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while(left < right){
            int temp = nums[left] + nums[right];
            if(temp == target) return new int[]{nums[left], nums[right]};
            if(temp > target) right--;
            else left++;
        }
        return new int[0];
    }
}
```

# 55.和为s的连续正数序列-简单

> 题目

输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

> 思路：双指针：滑动窗口

> 代码

```java
class Solution {
    public int[][] findContinuousSequence(int target) {
        int left = 1, right = 1, sum = 0;
        List<int[]> list = new ArrayList<>();
        while(left <= target / 2){
            if(sum == target){
                int[] temp = new int[right - left];
                for(int i = left; i < right; i++) temp[i - left] = i;
                list.add(temp);
            }
            if(sum < target){
                sum += right;
                right++;
            }else{
                sum -= left;
                left++;
            }
        }
        return list.toArray(new int[0][0]);
    }
}
```

# 55.翻转单词顺序-简单

> 题目

输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

> 思路：双指针

> 代码

```java
class Solution {
    public String reverseWords(String s) {
        String trim = s.trim();
        if(trim.length() == 0) return "";
        StringBuffer res = new StringBuffer();
        char[] str = trim.toCharArray();
        int left = str.length - 1, right = str.length;
        while(left >= 0){
            //遍历到单词末尾，也就是找到第一个空格
            while(left >= 0 && str[left] != ' ') left--;
            //两个指针间的字符串即是一个单词
            res.append(trim.substring(left + 1, right));
            res.append(" ");
            //跳过后续多余的空格
            while(left >= 0 && str[left] == ' ') left--;
            //右指针指向下一个单词的末尾字符位置+1
            right = left + 1;
        }
        return res.substring(0, res.length() - 1);
    }
}
```

> 思路：使用split()

> 代码

```java
class Solution {
    public String reverseWords(String s) {
        String trim = s.trim();
        if(trim.length() == 0) return "";
        String[] strArr = trim.split(" ");
        StringBuffer res = new StringBuffer();
        for(int i = strArr.length - 1; i >= 0; i--){
            if(!strArr[i].equals("")){
                // System.out.println(strArr[i]);
                res.append(strArr[i]);
                res.append(" ");
            }
        }
        return res.substring(0, res.length() - 1);
    }
}
```

# 55. 左旋转字符串-简单

> 题目

字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

> 思路：使用substr()或自己字符拼接

> 代码

```java
class Solution {
    public String reverseLeftWords(String s, int n) {
        int len = s.length();
        if(len == 0) return "";
        n = n % len;
        if(n == 0) return s;
        StringBuffer res = new StringBuffer();
        res.append(s.substring(n, len));
        res.append(s.substring(0, n));
        return res.toString();
    }
}
```

```java
class Solution {
    public String reverseLeftWords(String s, int n) {
        int len = s.length();
        if(len == 0) return "";
        n = n % len;
        if(n == 0) return s;
        StringBuffer res = new StringBuffer();
        for(int i = n; i < n + len; i++){
            //这样取余相当于循环了一遍
            res.append(s.charAt(i % len));
        }
        return res.toString();
    }
}
```

# 55. 滑动窗口的最大值-困难

> 题目

给定一个数组 `nums` 和滑动窗口的大小 `k`，请找出所有滑动窗口里的最大值。

> 思路1：暴力

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

> 思路2：单调队列

维护一个队头到队尾单调递减的双端队列，每到新窗口时即下一个新元素入队时：1、首先得判断现在的队列头元素是否是上一个窗口的第一个值，如果是就需要移除这个队头元素。2、维持单调队列

> 代码

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int len = nums.length;
        if(len == 0 || k > len) return new int[0];
        int[] res = new int[len - k + 1];
        Deque<Integer> deque = new LinkedList<>();
        //先将第一个窗口入单调队列
        for(int i = 0; i < k; i++){
            while(!deque.isEmpty() && deque.peekLast() < nums[i]){
                deque.removeLast();
            }
            deque.addLast(nums[i]);
        }
        res[0] = deque.peekFirst();
        for(int i = k; i < nums.length; i++){
            //如果此时最大值(即队列头)是上一个窗口的第一个元素那么需要移除掉
            if(deque.peekFirst() == nums[i - k]){
                deque.removeFirst();
            }
            //维护单调队列
            while(!deque.isEmpty() && deque.peekLast() < nums[i]){
                deque.removeLast();
            }
            deque.addLast(nums[i]);
            //最大值放入结果数组中
            res[i - k + 1] = deque.peekFirst();
        }
        return res;
    }
}
```

# 55. 队列的最大值-中等

> 题目

请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。

若队列为空，pop_front 和 max_value 需要返回 -1

> 思路1：维护一个普通队列和单调递减的双端队列

注意在判断出队元素与单调队列头元素是否相等时不要用 “ == ”，必须用equals，因为存的是Integer，不能直接比，数值较小也能用==是因为Integer的自动装箱时用的提前缓存好的Integer对象，同一个对象所有==也能相等。

> 代码

```java
class MaxQueue {

    Queue<Integer> queue = new LinkedList<>();
    Deque<Integer> deque = new LinkedList<>();
    public MaxQueue() {}
    
    public int max_value() {
        return deque.isEmpty() ? -1 : deque.peekFirst();
    }
    
    public void push_back(int value) {
        queue.offer(value);
        while(!deque.isEmpty() && deque.peekLast() < value)
            deque.removeLast();
        deque.addLast(value);
    }
    
    public int pop_front() {
        if(queue.isEmpty()) return -1;
        if(queue.peek().equals(deque.peekFirst()))
            deque.removeFirst();
        return queue.poll();
    }
}
```

# 56. n个骰子的点数-中等

> 题目

把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率

> 思路：动态规划（自顶向下）

状态：n枚骰子一共有6^n种可能，函数dicesProbability(int n, int sum)表示n枚骰子总共点数和为sum的可能有多少种。

计算出每个和的可能后直接和总可能数相比就OK了！

> 代码

```java
class Solution {
    int[][] dp = new int[12][70];
    public double[] dicesProbability(int n) {
        double[] res = new double[5 * n + 1];
        int all = (int)Math.pow(6, n);
        //和的范围是：[n, 6n]
        for(int i = n, j = 0; i <= 6 * n; i++, j++){
            res[j] = dicesProbability(n, i) * 1.0 / all;
        }
        return res;
    }
    //n枚骰子一共有6^n种可能，这函数表示n枚骰子总共点数和为sum的可能有多少种
    public int dicesProbability(int n, int sum) {
        if(sum < n || sum > 6 * n) return 0;
        if(n == 1) return 1;
        int res = 0;
        for(int i = 1; i <= 6 && i < sum; i++){
            if(dp[n-1][sum-i] != 0){
                res += dp[n-1][sum-i];
            }
            else{
                dp[n-1][sum-i] = dicesProbability(n-1, sum - i);
                res += dp[n-1][sum-i];
            }
        }
        return res;
    }
}
```

> 思路：动态规划（自底向上）

状态：dp[i\][j]表示i枚骰子总共点数和为j的可能有多少种。

转移方程：dp[i\][j] = dp[i - 1\][j - k]。其中k为[1, 6]。含义是拿一个骰子出来，它出现的数字可能性（一个骰子只有一种可能）加上剩余i-1枚骰子数字的和为：减去拿出来的一枚骰子的数的可能性

空间优化：由于只用到了上一个阶段的数据（n-1枚骰子）所以可以不用把全部的数据都记录着

**空间优化时注意点：**

1. 要考虑拿出一枚骰子后剩余的总数和最小为i-1枚骰子都为1时
2. 需要额外一个数组保留上一阶段的值：因为如果只使用一个数组的话填充新的值时会覆盖上个阶段的值，后面再用到上个阶段的值时就用成了这个阶段新填充的值。
3. 新填充时需要初始化为0：因为保存着上一个阶段的值
4. 如果这一阶段从后面开始填充就不会覆盖那就不需要额外的一个数组

> 代码

```java
class Solution {
    public double[] dicesProbability(int n) {
        //由于只用到了上一个阶段的数据（n-1枚骰子）所以可以不用把全部的数据都记录着
       int[] predp = new int[70];
       int[] nowdp = new int[70];
       //初始化第一阶段（只有一枚骰子时）每一种和的可能性。
       for(int i = 1; i <= 6; i++){
           predp[i] = 1;
       }
       //第一重循环：骰子枚数是[1, n],1枚时已经初始化过
       for(int i = 2; i <= n; i++){
           //第二重循环：和的范围是[骰子枚数, 6 * 骰子枚数]
           for(int j = i; j <= 6 * i; j++){
               //第三重循环：拿出来 的一枚骰子的可能的数字,其数字除了在1-6还不能大于 “和”
               //新填充时需要初始化为0
               nowdp[j] = 0;
               for(int k = 1; k <= 6; k++){
                //dp[i][j] += dp[i - 1][j - k];
                //总数和必须大于等于上一个阶段(i-1枚骰子时)的最小和即i-1枚骰子都为1
                 if (j - k < i-1) {
                        break;
                    }
                    nowdp[j] += predp[j - k];
               }
            //    System.out.println("j = " + j + ",dp[j] = " + dp[j]);
           }
           int[] temp = nowdp;
           nowdp = predp;
           predp = temp;
       }
       double[] res = new double[5 * n + 1];
       int all = (int)Math.pow(6, n);
       for(int i = n, j = 0; i <= 6 * n; i++, j++){
           res[j] = predp[i] * 1.0 / all;
       }
        return res;
    }
}
```

# 56.  圆圈中最后剩下的数字(约瑟夫环)-中等

> 题目

0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

> 思路1：动态规划（自顶向下）

[大佬题解](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/solution/jian-zhi-offer-62-yuan-quan-zhong-zui-ho-dcow/)

f(n, m)表示[0, n-1]，间隔为m，从第一个数开始，返回最后剩下的数下标。

主要是分析f(n, m)怎么根据子问题f(n-1, m)的结果得到答案。例如n=5，m=3：

f(n, m):求0,1,2,3,4从第一个数开始，间隔为3，最后剩下的一个数，我们看看它执行了一轮后也就是删除了第一个该删除的元素，这里是删除2，下一次该从3开始，相当于：求3,4,0,1从第一个数开始，间隔为3，最后剩下的一个数

即：求0,1,2,3,4从第一个数开始，间隔为3，最后剩下的一个数  `等价于`  求3,4,0,1从第一个数开始，间隔为3，最后剩下的一个数

再看f(n-1, m) = 求0,1,2,3从第一个数开始，间隔为3，最后剩下的一个数 。我们对比下f(n, m)：

f(n, m)  =   求3,4,0,1从第一个数开始，间隔为3，最后剩下的一个数

f(n-1, m) = 求0,1,2,3从第一个数开始，间隔为3，最后剩下的一个数

通过观察+运气+天才找规律得出f(n-1, m)的数与f(n, m)的数存在对应关系：f(n, m) = (f(n-1, m) + m % n) % n。

> 代码

```java
public int lastRemaining2(int n, int m) {
    	//边界：约瑟夫环长度为1，剩下的就是下标0
        if(n == 1) return 0;
        int res= lastRemaining(n - 1, m);
        return (res + m) % n;
    }
```

> 思路2：动态规划（自底向上）

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

> 思路3：常规思路：循环数组或链表

> 代码

```java
class Solution {
    public int lastRemaining(int n, int m) {
        ArrayList<Integer> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            list.add(i);
        }
        int idx = 0;
        while (n > 1) {
            //下一个开始的数就是该下标，因为ArrayList删除后是将后面的向前移动
            idx = (idx + m - 1) % n;
            list.remove(idx);
            n--;
        }
        return list.get(0);
    }
}
```

# 56.  股票的最大利润-中等

> 题目

假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

> 思路1：动态规划

状态：dp[i]表示第i天卖出能获得的最大利润。

转移方程：dp[i] = dp[i] + price[i] - price[i-1]。price[i]是第i天的股票价格

空间优化：由于只需要前一天卖出的最大利润，所以只需一个变量保存即可。

> 代码

```java
class Solution {
    public int maxProfit(int[] prices) {
        if(prices.length == 0) return 0;
        int curMax = 0, max = 0;
        for(int i = 1; i < prices.length; i++){
            curMax += prices[i] - prices[i - 1];
            if(curMax < 0) curMax = 0;
            if(curMax > max) max = curMax;
        }
        return max;
    }
}
```

# 56.  求1+2+…+n-中等

> 题目

求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

> 思路1：异常代替判断

虽然递归代替了循环，因为不能if，还要考虑递归到n == 0时怎么停止递归。

> 代码

```java
class Solution {
    public int sumNums(int n) {
        try{
            return 1 / n + n + sumNums(n - 1);
        }catch(Exception e){
            return -1;
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

# 56.  不用加减乘除做加法-中等

> 题目

写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

> 思路:位运算

- ^ 亦或 ：相当于 无进位的求和。 想象10进制下的模拟情况：（如:19+1=20；无进位求和就是10，而非20；因为它不管进位情况）

- & 与 ：相当于求每位的进位数。先看定义：1&1=1；1&0=0；0&0=0；即都为1的时候才为1，正好可以模拟进位数的情况,还是想象10进制下模拟情况：（9+1=10，如果是用&的思路来处理，则9+1得到的进位数为1，而不是10，所以要用<<1向左再移动一位，这样就变为10了）；

这样公式就是：（a^b) ^ ((a&b)<<1) 即：每次无进位求 + 每次得到的进位数--------我们需要不断重复这个过程，直到进位数为0为止；

> 代码

```java
class Solution {
    public int add(int a, int b) {
        int curBit = a ^ b, carryBit = (a & b) << 1;
        //进位如果是0表示就不需要进了，当前位就是结果了，否则还需要继续计算下一个当前位和进位
        while(carryBit != 0){
            //进位不为0，根据现在的当前位和进位求出下一个当前位和进位
            int pre = carryBit;
            carryBit = (curBit & carryBit) << 1;//下一个进位
            curBit = curBit ^ pre;//下一个当前位
        }
        return curBit;
    }
}
```

# 57.  不用加减乘除做加法-中等



> 题目

写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

> 思路:位运算

- ^ 亦或 ：相当于 `无进位的求和`。 想象10进制下的模拟情况：（如:19+1=20；无进位求和就是10，而非20；因为它不管进位情况）

- & 与 ：相当于`求每位的进位数`。先看定义：1&1=1；1&0=0；0&0=0；即都为1的时候才为1，正好可以模拟进位数的情况,还是想象10进制下模拟情况：（9+1=10，如果是用&的思路来处理，则9+1得到的进位数为1，而不是10，所以`要用<<1向左再移动一位`，这样就变为10了）；

这样公式就是：（a^b) ^ ((a&b)<<1) 即：每次无进位求 + 每次得到的进位数--------我们需要不断重复这个过程，直到进位数为0为止；

> 思路:位运算

- ^ 亦或 ：相当于 无进位的求和。 想象10进制下的模拟情况：（如:19+1=20；无进位求和就是10，而非20；因为它不管进位情况）

- & 与 ：相当于求每位的进位数。先看定义：1&1=1；1&0=0；0&0=0；即都为1的时候才为1，正好可以模拟进位数的情况,还是想象10进制下模拟情况：（9+1=10，如果是用&的思路来处理，则9+1得到的进位数为1，而不是10，所以要用<<1向左再移动一位，这样就变为10了）；

这样公式就是：（a^b) ^ ((a&b)<<1) 即：每次无进位求 + 每次得到的进位数--------我们需要不断重复这个过程，直到进位数为0为止；

> 代码

```java
class Solution {
    public int strToInt(String str) {
       String s = str.trim();
        char[] arr = s.toCharArray();
        long res = 0;
        int flag = 1;
        for(int i = 0; i < arr.length; i++){
            //不能放外面判断，因为最后循环外面直接返回的没再判断了，可能最后一次
            //循环完刚好超过了范围但又不会在进入循环进行判断了。 
            //if(res * flag > Integer.MAX_VALUE) return Integer.MAX_VALUE;
            //if(res * flag < Integer.MIN_VALUE) return Integer.MIN_VALUE;
            if(arr[i] >= '0' && arr[i] <= '9'){
                res = res * 10 + (arr[i] - '0');
                if(res * flag > Integer.MAX_VALUE) return Integer.MAX_VALUE;
                if(res * flag < Integer.MIN_VALUE) return Integer.MIN_VALUE;
            }else{
                if(i == 0 && (arr[i] == '+' || arr[i] == '-')){
                    if(arr[i] == '-'){
                        flag *= -1;
                    }
                }else{
                    break;
                }
            }
        }
        res *= flag;
        return (int)res;
    }
}
```



# 59.  二叉搜索树的最近公共祖先-简单

> 题目

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

**说明:**

- 所有节点的值都是唯一的。
- p、q 为不同节点且均存在于给定的二叉树中。

> 思路:递归

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



# 60.  II. 二叉树的最近公共祖先-简单

> 题目

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

**说明:**

- 所有节点的值都是唯一的。
- p、q 为不同节点且均存在于给定的二叉树中。

> 思路:递归

如果该结点左右子树都分别存在p或q结点（所有节点的值都是唯一的。）那么这个结点就是p、q的最近祖先。

如果只有一边存在那么p、q最近祖先就是最前面（最上面）那个等于p或q的结点，所以用的前序遍历

都不存在返回null。

> 代码

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        //找到左右子树中存在p、q的结点
        if(root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        //两边都没找到，题目说p、q一定存在，所以也可以不用加这句判断边界
        if(left == null && right == null) return null;
        //只有左右子树其中一边存在就返回存在的结点，这是p、q其中一个就是它们的祖先
        //注意：这样如果给了两个结点一个存在一个不存在树中那么也会返回存在树中的这个结点，但
        //题目说了p、q一定存在所有就没关系
        if(left == null) return right;
        if(right == null) return left;
        //左右子树都存在值时当前节点就是双方祖先
        return root;
    }
}
```

# 60.  I构建乘积数组-简单

> 题目

- 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B[i] 的值是数组 A 中除了下标 i 以外的元素的积, 即 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。


> 思路:上三角、下三角

根据表格的主对角线（全为 11 ），可将表格分为 **上三角** 和 **下三角** 两部分。分别迭代计算下三角和上三角两部分的乘积，即可 **不使用除法** 就获得结果。

> 代码

```java
class Solution {
    public int[] constructArr(int[] a) {
        if(a.length == 0) return new int[0];
        int[] left = new int[a.length], right = new int[a.length];
        left[0] = right[a.length - 1] = 1;
        //下三角每一行的乘积值
        for(int i = 1; i < a.length; i++){
            //前一个的积乘以a当前的前一个
            left[i] = left[i - 1] * a[i - 1];
        }
        //上三角每一行的乘积值
        for(int i = a.length - 2; i >= 0; i--){
            right[i] = right[i + 1] * a[i + 1];
        }
        //当前值等于左边乘右边
        int[] res = new int[a.length];
        for(int i = 0; i < a.length; i++){
            res[i] = left[i] * right[i];
        }
        return res;
    }
}
```
