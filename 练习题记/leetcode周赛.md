# 第 237 场力扣周赛

## 1、判断句子是否为全字母句

> 题目

全字母句 指包含英语字母表中每个字母至少一次的句子。

给你一个仅由小写英文字母组成的字符串 sentence ，请你判断 sentence 是否为 全字母句 。

如果是，返回 true ；否则，返回 false 。

> 思路

> 代码

```java
class Solution {
    public boolean checkIfPangram(String sentence) {
        if(sentence.length() < 26) return false;
        HashSet<Character> set = new HashSet<>();
        for(int i = 0; i < 26; i++){
            set.add((char)('a' + i));
        }
        for(int i = 0; i < sentence.length(); i++){
            if(set.contains(sentence.charAt(i)))
                set.remove(sentence.charAt(i));
        }
        return set.isEmpty();
    }
}
```

## 2、 雪糕的最大数量

> 题目

夏日炎炎，小男孩 Tony 想买一些雪糕消消暑。

商店中新到 n 支雪糕，用长度为 n 的数组 costs 表示雪糕的定价，其中 costs[i] 表示第 i 支雪糕的现金价格。Tony 一共有 coins 现金可以用于消费，他想要买尽可能多的雪糕。

给你价格数组 costs 和现金量 coins ，请你计算并返回 Tony 用 coins 现金能够买到的雪糕的 最大数量 。

注意：Tony 可以按任意顺序购买雪糕。

> 思路：贪心

排好序一直搞就OK

> 代码

```java
class Solution {
    public int maxIceCream(int[] costs, int coins) {
        int res = 0, sum = 0;
        Arrays.sort(costs);
        for(int i = 0; i < costs.length; i++){
            sum += costs[i];
            if(sum > coins) return res;
            res++;
        }
        return res;
    }
}
```

## 3、单CPU执行

> 题目



> 思路：



> 代码

## 4、所有数对按位与结果的异或和

> 题目

列表的 异或和（XOR sum）指对所有元素进行按位 XOR 运算的结果。如果列表中仅有一个元素，那么其 异或和 就等于该元素。

例如，[1,2,3,4] 的 异或和 等于 1 XOR 2 XOR 3 XOR 4 = 4 ，而 [3] 的 异或和 等于 3 。
给你两个下标 从 0 开始 计数的数组 arr1 和 arr2 ，两数组均由非负整数组成。

根据每个 (i, j) 数对，构造一个由 arr1[i] AND arr2[j]（按位 AND 运算）结果组成的列表。其中 0 <= i < arr1.length 且 0 <= j < arr2.length 。

返回上述列表的 异或和 。

> 思路：找规律，根据异或运算的结合律、交换律两个性质和异或运算与按位与运算的分配率

给两个数组a1、a2......an和b1、b2......bn。这个题让我们计算：

(a1 & b1) ^ (a1 & b2) ^ ...... ^ (a1 & bn) ^ 

(a2 & b1) ^ (a2 & b2) ^ ...... ^ (a2 & bn) ^

......

(an & b1) ^ (an & b2) ^ ...... ^ (an & bn) = ？

其实就是和我们以前做数学时遇到的那种找规律求和的题一样。光想可能不好找出规律，但我们一列出上面的式子我们很容易看出来同一列似乎有某种规律。运用异或运算的交换律与结合律我们按列来计算，来看第一列：

(a1 & b1) ^ (a2 & b1) ^ ...... ^ (an & b1)

我们利用分配率可以转化为：b1 & (a1 ^ a2 ^ ...... ^ an)。这就是第一列也就是关于b1这一列，后面还有b2到bn，所以再算上后面的b2到bn就是：

{ b1 & (a1 ^ a2 ^ ...... ^ an) } ^ { b2 & (a1 ^ a2 ^ ...... ^ an) } ^ ...... ^ { bn & (a1 ^ a2 ^ ...... ^ an) }

再转化一下就是：(b1 ^ b2 ^ ...... ^ bn) & (a1 ^ a2 ^ ...... ^ an)

所以最终：

(a1 & b1) ^ (a1 & b2) ^ ...... ^ (a1 & bn) ^ 

(a2 & b1) ^ (a2 & b2) ^ ...... ^ (a2 & bn) ^

......

(an & b1) ^ (an & b2) ^ ...... ^ (an & bn) =  (b1 ^ b2 ^ ...... ^ bn) & (a1 ^ a2 ^ ...... ^ an)

总结起来看：

原式 = (a1 & b1) ^ (a1 & b2) ^ ...... ^ (a1 & bn) ^ 

​			(a2 & b1) ^ (a2 & b2) ^ ...... ^ (a2 & bn) ^

​			......

​			(an & b1) ^ (an & b2) ^ ...... ^ (an & bn) 

​		= (a1 & b1) ^ (a2 & b1) ^ ...... ^ (an & b1) ^ 

​			(a1 & b2) ^ (a2 & b2) ^ ...... ^ (a2 & b1) ^ 

​			......

​			(a1 & bn) ^ (a2 & bn) ^ ...... ^ (an & bn)

​		= { b1 & (a1 ^ a2 ^ ...... ^ an) } ^ { b2 & (a1 ^ a2 ^ ...... ^ an) } ^ ...... ^ { bn & (a1 ^ a2 ^ ...... ^ an) }

​		= (b1 ^ b2 ^ ...... ^ bn) & (a1 ^ a2 ^ ...... ^ an)

> 代码

```java
class Solution {
    public int getXORSum(int[] arr1, int[] arr2) {
        int res1 = 0, res2 = 0;
        for(int num: arr1){
            res1 ^= num;
        }
        for(int num: arr2){
            res2 ^= num;
        }
        return res1 & res2;
    }
}
```

