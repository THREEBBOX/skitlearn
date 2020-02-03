class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def uniquePaths(self, m: int, n: int) -> int:

        if m < 2:
            return 1
        if n < 2:
            return 1
        memory = [[-1] * (n + 1)] * (m + 1)
        print(memory)
        result = self.getMethod(m, n, memory)
        print(memory)
        return result

    def getMethod(self, m, n, memory):

        if m==1 or n==1:
            memory[m][n]=1
            return 1
        if memory[m][n]!=-1:
            print("heh")
            return memory[m][n]
        if memory[m-1][n]!=-1:
            upm=memory[m-1][n]
        else:
            upm=self.getMethod(m-1,n,memory)
        if memory[m][n-1]!=-1:
            upn=memory[m][n-1]
        else:
            upn=self.getMethod(m,n-1,memory)
        return upm+upn
if __name__ == '__main__':
    s=Solution()
    print("reslult",s.uniquePaths(3,7))

