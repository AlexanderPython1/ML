import sys

def solve():
    s = sys.stdin.read().strip()
    if not s:
        print(0)
        return

    n = len(s)
    digits = [int(c) for c in s]

    pref = [[0] * 3 for _ in range(n + 1)]
    for i in range(n):
        r = digits[i] % 3
        pref[i + 1][0] = pref[i][0]
        pref[i + 1][1] = pref[i][1]
        pref[i + 1][2] = pref[i][2]
        pref[i + 1][r] += 1

    def count_left(pos, r):
        return pref[pos][r]

    best_len = 0

   
    zero_count = s.count('0')
    if zero_count > 0:
        best_len = max(best_len, zero_count)

    
    pairs = [(0, 0), (2, 5), (5, 0), (7, 5)]

    for x, y in pairs:
        
        j = -1
        for idx in range(n - 1, -1, -1):
            if digits[idx] == y:
                j = idx
                break
        if j == -1:
            continue

        i = -1
        for idx in range(j - 1, -1, -1):
            if digits[idx] == x:
                i = idx
                break
        if i == -1:
            continue

        
        sum_pair = x + y
        rem_pair = sum_pair % 3

      
        c0 = count_left(i, 0)
        c1 = count_left(i, 1)
        c2 = count_left(i, 2)

        base_len = 2 + c0  

    

        max_add = 0
        start1 = max(0, c1 - 3)
        start2 = max(0, c2 - 3)

        for take1 in range(start1, c1 + 1):
            for take2 in range(start2, c2 + 1):
                if (take1 + 2 * take2 + rem_pair) % 3 == 0:
                    max_add = max(max_add, take1 + take2)

        total_len = base_len + max_add
        best_len = max(best_len, total_len)

   
    print(n - best_len)

if __name__ == "__main__":
    solve()